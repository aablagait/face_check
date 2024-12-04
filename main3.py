import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk
import os
from collections import Counter
import numpy as np
from deepface import DeepFace
from collections import defaultdict
import csv


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


# === Интерфейс программы ===

class ImageClassifierApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Classifier")
        self.root.geometry("800x600")
        self.root.resizable(False, False)

        self.embeddings = load_embeddings("embeddings.csv")  # Укажите путь к вашему CSV файлу
        self.mode = tk.IntVar(value=0)  # 0 - Одно изображение, 1 - Папка

        self.create_widgets()

    def create_widgets(self):
        # Верхняя часть интерфейса
        mode_frame = tk.Frame(self.root)
        mode_frame.pack(pady=10)

        tk.Label(mode_frame, text="Mode:", font=("Arial", 12)).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(mode_frame, text="Single Image", variable=self.mode, value=0).pack(side=tk.LEFT)
        ttk.Radiobutton(mode_frame, text="Folder", variable=self.mode, value=1).pack(side=tk.LEFT)

        # Кнопки выбора файла/папки
        self.select_button = ttk.Button(self.root, text="Select Image/Folder", command=self.select_input)
        self.select_button.pack(pady=20)

        # Отображение выбранного изображения
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=20)

        # Результаты
        self.results_frame = tk.Frame(self.root)
        self.results_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.results_text = tk.Text(self.results_frame, wrap=tk.WORD, state=tk.DISABLED, font=("Arial", 12))
        self.results_text.pack(fill=tk.BOTH, expand=True)

    def select_input(self):
        if self.mode.get() == 0:
            file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
            if file_path:
                self.classify_single_image(file_path)
        else:
            folder_path = filedialog.askdirectory()
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
            messagebox.showerror("Error", "Failed to classify the image.")

    def classify_folder(self, folder_path):
        most_common_class, classifications = process_images(self.embeddings, folder_path)
        if classifications:
            self.display_results(f"Most common type: {most_common_class}\n\nDetails:\n" +
                                 "\n".join(f"{idx+1}. {cls}" for idx, cls in enumerate(classifications)))
        else:
            messagebox.showerror("Error", "No images were classified successfully.")

    def display_image(self, file_path):
        image = Image.open(file_path).resize((200, 200))
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo)
        self.image_label.image = photo

    def display_results(self, text):
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, text)
        self.results_text.config(state=tk.DISABLED)


# === Запуск приложения ===
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()
