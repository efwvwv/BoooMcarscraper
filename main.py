"""
Car Listing Scraper & Anonymizer — MVP
Автор: iMe AI Expert
Поддержка: AutoScout24 (базово), общий движок для Otomoto / AUTO.RIA будет добавлен следующим шагом.

Что делает:
1. Принимает ссылку на объявление.
2. Качает фото (через og:image / img теги) + текстовое описание.
3. Обрезает верх/низ фото (watermark) на заданный процент.
4. Ищет прямоугольные объекты похожие на номерной знак и размывает их.
5. Сохраняет всё в папку: photos/ + description.txt

Зависимости: requirements.txt
Запуск: python main.py
Сборка в exe: pyinstaller --noconfirm --onefile --windowed --name CarScraper main.py
"""

import sys
import os
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QSpinBox, QCheckBox, QTextEdit, QGroupBox,
    QProgressBar, QMessageBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}


def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', "_", name).strip()
    return name[:80] if name else "listing"


def fetch_page(url: str):
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def extract_title(soup: BeautifulSoup) -> str:
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        return og_title["content"]
    if soup.title:
        return soup.title.get_text(strip=True)
    return "listing"


def extract_description(soup: BeautifulSoup) -> str:
    parts = []
    og_desc = soup.find("meta", property="og:description")
    if og_desc and og_desc.get("content"):
        parts.append(og_desc["content"])

    # Попытка вытащить весь видимый текстовый блок описания (общий эвристический сбор)
    candidates = soup.find_all(["p", "div", "li"], limit=300)
    seen = set()
    collected = []
    for c in candidates:
        text = c.get_text(" ", strip=True)
        if text and len(text) > 40 and text not in seen:
            seen.add(text)
            collected.append(text)
        if len(collected) > 40:
            break

    full_text = "\n".join(parts + collected)
    return full_text if full_text.strip() else "Описание не найдено автоматически."


def extract_images(soup: BeautifulSoup, base_url: str):
    urls = set()

    for tag in soup.find_all("meta", property="og:image"):
        if tag.get("content"):
            urls.add(tag["content"])

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-srcset")
        if not src:
            continue
        if src.startswith("//"):
            src = "https:" + src
        elif src.startswith("/"):
            parsed = urlparse(base_url)
            src = f"{parsed.scheme}://{parsed.netloc}{src}"
        if src.startswith("http") and any(ext in src.lower() for ext in [".jpg", ".jpeg", ".png", ".webp"]):
            urls.add(src)

    return list(urls)


def download_image(url: str, path: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        return True
    except Exception:
        return False


def crop_watermark(image_path: str, top_pct: int, bottom_pct: int):
    img = cv2.imread(image_path)
    if img is None:
        return
    h, w = img.shape[:2]
    top = int(h * top_pct / 100)
    bottom = int(h * (1 - bottom_pct / 100))
    if bottom <= top:
        return
    cropped = img[top:bottom, :]
    cv2.imwrite(image_path, cropped)


def blur_plates(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    h_img, w_img = img.shape[:2]

    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        aspect = w / float(h)
        area_ratio = (w * h) / float(w_img * h_img)
        # Номерной знак: вытянутый прямоугольник, не слишком большой и не крошечный
        if 2.0 < aspect < 6.0 and 0.0008 < area_ratio < 0.04:
            roi = img[y:y + h, x:x + w]
            roi = cv2.GaussianBlur(roi, (31, 31), 0)
            img[y:y + h, x:x + w] = roi

    cv2.imwrite(image_path, img)


class WorkerThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, url, out_dir, top_pct, bottom_pct, do_blur):
        super().__init__()
        self.url = url
        self.out_dir = out_dir
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
        self.do_blur = do_blur

    def run(self):
        try:
            self.log.emit("Загружаю страницу...")
            soup = fetch_page(self.url)

            title = safe_filename(extract_title(soup))
            folder = os.path.join(self.out_dir, title)
            photos_dir = os.path.join(folder, "photos")
            os.makedirs(photos_dir, exist_ok=True)

            self.log.emit("Извлекаю описание...")
            description = extract_description(soup)
            with open(os.path.join(folder, "description.txt"), "w", encoding="utf-8") as f:
                f.write(f"Источник: {self.url}\n\n{description}")

            self.log.emit("Ищу фотографии...")
            images = extract_images(soup, self.url)
            if not images:
                self.log.emit("⚠️ Фото не найдены (возможно сайт грузит их через JS — нужна версия с Selenium).")

            total = len(images)
            for idx, img_url in enumerate(images, start=1):
                ext = os.path.splitext(urlparse(img_url).path)[1] or ".jpg"
                ext = ext.split("?")[0][:5]
                fname = f"photo_{idx}{ext}"
                fpath = os.path.join(photos_dir, fname)

                ok = download_image(img_url, fpath)
                if ok:
                    try:
                        crop_watermark(fpath, self.top_pct, self.bottom_pct)
                        if self.do_blur:
                            blur_plates(fpath)
                    except Exception as e:
                        self.log.emit(f"Ошибка обработки {fname}: {e}")
                    self.log.emit(f"✓ {fname} обработано")
                else:
                    self.log.emit(f"✗ Не удалось скачать {fname}")

                if total:
                    self.progress.emit(int(idx / total * 100))

            self.finished_ok.emit(folder)

        except Exception as e:
            self.failed.emit(str(e))


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Car Listing Scraper & Anonymizer")
        self.resize(640, 560)
        self.setStyleSheet("""
            QWidget { background-color: #1e1f29; color: #e8e8e8; font-size: 13px; }
            QLineEdit, QSpinBox, QTextEdit {
                background-color: #2b2d3a; border: 1px solid #3c3e4d;
                border-radius: 6px; padding: 6px; color: #fff;
            }
            QPushButton {
                background-color: #6c5ce7; border: none; border-radius: 8px;
                padding: 10px; color: white; font-weight: bold;
            }
            QPushButton:hover { background-color: #8075f0; }
            QPushButton:disabled { background-color: #45465a; }
            QGroupBox {
                border: 1px solid #3c3e4d; border-radius: 8px; margin-top: 10px; padding-top: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; color: #a29bfe; }
            QProgressBar {
                background-color: #2b2d3a; border-radius: 6px; text-align: center; color: white;
            }
            QProgressBar::chunk { background-color: #6c5ce7; border-radius: 6px; }
        """)
        self.out_dir = os.path.join(os.path.expanduser("~"), "Desktop", "CarScraperOutput")

        layout = QVBoxLayout()

        title = QLabel("🚗 Car Listing Scraper & Anonymizer")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        layout.addWidget(title)

        url_row = QHBoxLayout()
        self.url_input = QLineEdit()
        self.url_input.setPlaceholderText("Вставь ссылку на объявление (AutoScout24 / Otomoto / AUTO.RIA)...")
        url_row.addWidget(self.url_input)
        layout.addLayout(url_row)

        settings_box = QGroupBox("Настройки")
        settings_layout = QVBoxLayout()

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Обрезать сверху (%):"))
        self.top_spin = QSpinBox()
        self.top_spin.setRange(0, 40)
        self.top_spin.setValue(0)
        top_row.addWidget(self.top_spin)
        settings_layout.addLayout(top_row)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(QLabel("Обрезать снизу (%):"))
        self.bottom_spin = QSpinBox()
        self.bottom_spin.setRange(0, 40)
        self.bottom_spin.setValue(8)
        bottom_row.addWidget(self.bottom_spin)
        settings_layout.addLayout(bottom_row)

        self.blur_check = QCheckBox("Замазывать номерные знаки")
        self.blur_check.setChecked(True)
        settings_layout.addWidget(self.blur_check)

        folder_row = QHBoxLayout()
        self.folder_label = QLabel(self.out_dir)
        self.folder_label.setWordWrap(True)
        folder_btn = QPushButton("Папка сохранения")
        folder_btn.clicked.connect(self.choose_folder)
        folder_row.addWidget(self.folder_label, 1)
        folder_row.addWidget(folder_btn)
        settings_layout.addLayout(folder_row)

        settings_box.setLayout(settings_layout)
        layout.addWidget(settings_box)

        self.start_btn = QPushButton("Обработать объявление")
        self.start_btn.clicked.connect(self.start_processing)
        layout.addWidget(self.start_btn)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        layout.addWidget(self.log_box)

        self.setLayout(layout)

    def choose_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Выбери папку сохранения", self.out_dir)
        if folder:
            self.out_dir = folder
            self.folder_label.setText(folder)

    def start_processing(self):
        url = self.url_input.text().strip()
        if not url.startswith("http"):
            QMessageBox.warning(self, "Ошибка", "Вставь корректную ссылку на объявление.")
            return

        os.makedirs(self.out_dir, exist_ok=True)
        self.start_btn.setEnabled(False)
        self.progress.setValue(0)
        self.log_box.clear()

        self.worker = WorkerThread(
            url, self.out_dir,
            self.top_spin.value(), self.bottom_spin.value(),
            self.blur_check.isChecked()
        )
        self.worker.progress.connect(self.progress.setValue)
        self.worker.log.connect(self.append_log)
        self.worker.finished_ok.connect(self.on_finished)
        self.worker.failed.connect(self.on_failed)
        self.worker.start()

    def append_log(self, text):
        self.log_box.append(text)

    def on_finished(self, folder):
        self.start_btn.setEnabled(True)
        self.append_log(f"\n✅ Готово! Результат: {folder}")
        QMessageBox.information(self, "Готово", f"Обработка завершена.\nПапка:\n{folder}")

    def on_failed(self, error):
        self.start_btn.setEnabled(True)
        self.append_log(f"\n❌ Ошибка: {error}")
        QMessageBox.critical(self, "Ошибка", error)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
