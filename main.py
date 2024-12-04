import os
import csv
from collections import Counter, defaultdict

import numpy as np
from deepface import DeepFace
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PIL import Image
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QRadioButton, QFileDialog, QTextEdit, QMessageBox, QButtonGroup
)


# === Основные функции программы ===
def load_embeddings(csv_file):
    embeddings = defaultdict(list)
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            file_path = row[0]
            group = file_path.split(os.sep)[-2]
            vector = np.array(row[1:], dtype=float)
            embeddings[group].append(vector)
    return embeddings


def extract_embedding(image_path):
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name='Facenet', enforce_detection=False)
        return np.array(embedding[0]['embedding']) if embedding else None
    except Exception as e:
        print(f"Error processing input image: {e}")
        return None


def calculate_rmse(vector, group_vectors):
    diffs = [np.linalg.norm(vector - gv) for gv in group_vectors]
    return np.mean(diffs)


def classify_image(embeddings, input_image_path):
    input_vector = extract_embedding(input_image_path)
    if input_vector is None:
        return None

    scores = {}
    for group, group_vectors in embeddings.items():
        rmse = calculate_rmse(input_vector, group_vectors)
        scores[group] = rmse

    best_group = min(scores, key=scores.get)
    return best_group, scores


def process_images(embeddings, folder_path):
    classification_results = []

    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
            input_image_path = os.path.join(folder_path, file_name)
            result = classify_image(embeddings, input_image_path)
            if result:
                best_group, _ = result
                classification_results.append(best_group)

    if classification_results:
        most_common_class, count = Counter(classification_results).most_common(1)[0]
        return most_common_class, classification_results
    return None, []


# === Интерфейс приложения ===
class ImageClassifierApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Classifier")
        self.setGeometry(100, 100, 800, 600)

        self.embeddings = load_embeddings("embeddings.csv")  # Укажите путь к вашему CSV файлу
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Режим выбора
        mode_layout = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.single_image_radio = QRadioButton("Single Image")
        self.folder_radio = QRadioButton("Folder")
        self.single_image_radio.setChecked(True)

        self.mode_group.addButton(self.single_image_radio)
        self.mode_group.addButton(self.folder_radio)

        mode_layout.addWidget(QLabel("Mode:"))
        mode_layout.addWidget(self.single_image_radio)
        mode_layout.addWidget(self.folder_radio)

        layout.addLayout(mode_layout)

        # Кнопка выбора
        self.select_button = QPushButton("Select Image/Folder")
        self.select_button.clicked.connect(self.select_input)
        layout.addWidget(self.select_button)

        # Отображение изображения
        self.image_label = QLabel()
        self.image_label.setFixedSize(200, 200)
        self.image_label.setStyleSheet("border: 1px solid black;")
        # layout.addWidget(self.image_label, alignment=1)
        layout.addWidget(self.image_label, alignment=Qt.AlignCenter)

        # Результаты
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        layout.addWidget(self.results_text)

        self.setLayout(layout)

    def select_input(self):
        if self.single_image_radio.isChecked():
            file_path, _ = QFileDialog.getOpenFileName(
                self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)"
            )
            if file_path:
                self.classify_single_image(file_path)
        else:
            folder_path = QFileDialog.getExistingDirectory(self, "Select Folder")
            if folder_path:
                self.classify_folder(folder_path)

    def classify_single_image(self, file_path):
        # Отображение изображения
        self.display_image(file_path)

        # Классификация
        result = classify_image(self.embeddings, file_path)
        if result:
            best_group, scores = result
            self.display_results(f"Image belongs to: {best_group}\nScores:\n" +
                                 "\n".join(f"{group}: {score:.2f}" for group, score in scores.items()))
        else:
            QMessageBox.critical(self, "Error", "Failed to classify the image.")

    def classify_folder(self, folder_path):
        most_common_class, classifications = process_images(self.embeddings, folder_path)
        if classifications:
            self.display_results(f"Most common type: {most_common_class}\n\nDetails:\n" +
                                 "\n".join(f"{idx+1}. {cls}" for idx, cls in enumerate(classifications)))
        else:
            QMessageBox.critical(self, "Error", "No images were classified successfully.")

    def display_image(self, file_path):
        image = Image.open(file_path)
        image = image.resize((200, 200))
        image.save("temp_image.jpg")  # Сохраняем временное изображение для отображения
        pixmap = QPixmap("temp_image.jpg")
        self.image_label.setPixmap(pixmap)

    def display_results(self, text):
        self.results_text.setPlainText(text)


# === Запуск приложения ===
if __name__ == "__main__":
    app = QApplication([])
    window = ImageClassifierApp()
    window.show()
    app.exec_()
