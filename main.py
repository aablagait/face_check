import os
import sys
import csv
from collections import Counter, defaultdict
import numpy as np
from deepface import DeepFace
from tkinter import Tk, Label, Button, Radiobutton, StringVar, Text, filedialog, messagebox
from PIL import Image, ImageTk


if getattr(sys, 'frozen', False):
    # Если приложение запущено как исполняемый файл
    base_path = sys._MEIPASS
else:
    # Если приложение запущено из исходного кода
    base_path = os.path.dirname(__file__)

embeddings_file = os.path.join(base_path, 'embeddings.csv')


# === Основные функции программы ===
def load_embeddings(csv_file):
    embeddings = defaultdict(list)
    try:
        with open(csv_file, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)  # Пропускаем заголовок
            for row in reader:
                file_path = row[0]
                group = file_path.split('/')[-2]

                vector = np.array(row[1:], dtype=float)
                embeddings[group].append(vector)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
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
class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")
        self.root.geometry("800x600")

        self.embeddings = load_embeddings(embeddings_file)  # Укажите путь к вашему CSV файлу

        self.mode_var = StringVar(value="single")

        # Режим выбора
        mode_frame = Label(root)
        mode_frame.pack(pady=10)

        Label(mode_frame, text="Mode:").pack(side='left')
        Radiobutton(mode_frame, text="Загружать изображение", variable=self.mode_var, value="single").pack(side='left')
        Radiobutton(mode_frame, text="Папка", variable=self.mode_var, value="folder").pack(side='left')

        # Кнопка выбора
        self.select_button = Button(root, text="Выбор изображения/папки", command=self.select_input)
        self.select_button.pack(pady=10)

        # Отображение изображения
        self.image_label = Label(root)
        self.image_label.pack(pady=10)

        # Результаты
        self.results_text = Text(root, wrap='word', height=15)
        self.results_text.pack(pady=10)

    def select_input(self):
        if self.mode_var.get() == "single":
            # file_path = filedialog.askopenfilename(title="Select Image",
            #                                        filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
            file_path = filedialog.askopenfilename()
            if file_path:  # Проверяем выбран ли файл
                print(f"Selected image: {file_path}")  # Отладочное сообщение
                self.classify_single_image(file_path)
            else:
                print("No image selected.")  # Отладочное сообщение
                messagebox.showinfo("Info", "Не выбрано изображение.")  # Информируем пользователя
        else:
            folder_path = filedialog.askdirectory(title="Select Folder")
            if folder_path:
                print(f"Selected folder: {folder_path}")  # Отладочное сообщение
                self.classify_folder(folder_path)

    def classify_single_image(self, file_path):
        # Отображение изображения
        self.display_image(file_path)

        # Классификация
        result = classify_image(self.embeddings, file_path)

        if result:
            best_group, scores = result
            self.display_results(f"Изображение относится к группе: {best_group}\nРасстояние:\n" +
                                 "\n".join(f"{group}: {score:.2f}" for group, score in scores.items()))
            print(f"Classification result: {best_group}")  # Отладочное сообщение
        else:
            messagebox.showerror("Error", "Не удалось классифицировать изображение.")
            print("Classification failed.")  # Отладочное сообщение

    def classify_folder(self, folder_path):
        most_common_class, classifications = process_images(self.embeddings, folder_path)

        if classifications:
            self.display_results(f"Наиболее вероятный тип: {most_common_class}\n\nDetails:\n" +
                                 "\n".join(f"{idx + 1}. {cls}" for idx, cls in enumerate(classifications)))
            print(f"Folder classification result: {most_common_class}")  # Отладочное сообщение
        else:
            messagebox.showerror("Error", "Не удалось классифицировать ни одно изображение.")
            print("No images classified.")  # Отладочное сообщение

    def display_image(self, file_path):
        try:
            image = Image.open(file_path)

            # Если изображение в формате RGBA (с альфа-каналом), конвертируем в RGB
            if image.mode == 'RGBA':
                image = image.convert('RGB')

            image.thumbnail((200, 200))  # Изменение размера с сохранением пропорций

            # Сохранение ссылки на изображение для отображения в метке
            self.tk_image = ImageTk.PhotoImage(image)

            self.image_label.config(image=self.tk_image)  # Обновление метки с изображением
            self.image_label.image = self.tk_image  # Сохранение ссылки на изображение

            print("Image displayed successfully.")  # Отладочное сообщение

        except Exception as e:
            print(f"Error displaying image: {e}")
            messagebox.showerror("Error", "Не удалось отобразить изображение.")

    def display_results(self, text):
        self.results_text.delete(1.0, 'end')  # Очистка предыдущих результатов
        self.results_text.insert('end', text)  # Вставка новых результатов


# === Запуск приложения ===
if __name__ == "__main__":
    try:
        root = Tk()

        app = ImageClassifierApp(root)

        root.mainloop()

    except Exception as e:
        print(f"Application error: {e}")
