import os
import sys
import csv
from collections import Counter, defaultdict
import numpy as np
from deepface import DeepFace
from tkinter import Tk, Label, Button, Radiobutton, StringVar, Text, filedialog, messagebox, Toplevel
from PIL import Image, ImageTk



if getattr(sys, 'frozen', False):
    # Если приложение запущено как исполняемый файл
    base_path = sys._MEIPASS
else:
    # Если приложение запущено из исходного кода
    base_path = os.path.dirname(__file__)

embeddings_file = os.path.join(base_path, 'embeddings.csv')


ENFJ_Gamlet = 'главный герой. Умеет общаться со всеми, открытый, преданный, целеустремленный. Наделён харизмой.'
ENFP_Geksli = 'чемпион. Творческие, активные люди с сильно развитой интуицией. Быстро добиваются успеха в делах, но также быстро охладевают к ним.'
ENTJ_Djeck_London = 'командир. Рациональный тип личности с лидерскими качествами. Ответственный, инициативный, любит структурировать. Ошибки и неудачи воспринимает как возможность стать лучше. Решения принимает быстро, но взвешивает все риски.'
ENTP_Don_kihot = 'спорщик. Любят разговоры на интеллектуальные темы, сомневается в правоте слов собеседника, если они не доказаны логикой. Предпочитают факты, доказательства. Творческие, смелые, энергичные люди.'
ESFJ_Gygo = 'воспитатель. Ему необходимо общение. Коммуникабельность, способность выстраивать контакт со всеми делают его популярным. Ответственный, отзывчивый, неконфликтный, споры пытается уладить мирным путем.'
ESFP_Napoleon = 'артист. Дружелюбные, щедрые, заботливые люди. Легко привлекают к себе внимание окружающих, потому что с ними комфортно общаться и находиться рядом. Любят учиться и учить других.'
ESTJ_Shteerlic = 'режиссёр. Организованный, честный с собой и окружающими, к нему часто обращаются за помощью. И он с радостью помогает. Трудолюбивый, ответственный, ему можно доверить организацию важных мероприятий.'
ESTP_Jukov = 'делец, предприниматель. Азартный, увлечённый человек, любит рискованные мероприятия. Эффективен в проектах, где нужно действовать быстро и решительно.'
INFJ_Dostoevsky = 'адвокат, консультант, видящий. Редкая группа людей, составляющая меньше 1% населения. Это рациональные, прилежные, творческие личности, которые обладают прекрасной интуицией, внимательно относятся к окружающим.'
INFP_Esenin = 'посредник, целитель, медиатор. Уравновешенный, творческий человек с развитой системой ценностей. Идеалист, дружелюбный всегда, даже в конфликтах. Обладает глубоким чувством эмпатии.'
INTJ_Robesper = 'архитектор, стратег. Таким людям лучше всего работается в одиночестве. Они хорошо организованы, уверены в себе, строго соблюдают сроки работы. Объективны, ценят точность, систематичность, стремятся понять глубинные смыслы.'
INTP_Balzack = 'мыслитель, ученый. Компетентный, действует по алгоритму, быстро находит ошибки и несоответствия. Мыслитель хорошо работает в одиночестве, сдержан в своих эмоциях, предан близким людям.'
ISFJ_Dryzer = 'защитник, хранитель. Этот тип личности стремится к уединению, черпает энергию из внутреннего мира и очень сдержан в общении. Преданные, внимательные люди, ответственно относящиеся к своей работе. Они придерживаются правил, способны изменить эффективность работы команды в лучшую сторону.'
ISFP_Dyma = 'художник. Интроверт, любит одиночество, предпочитает глубокое общение с глазу на глаз. Избегает конфликтов, дружелюбный, создаёт положительную атмосферу вокруг себя.'
ISTJ_Gorkiy = 'логист, администратор, инспектор. Люди этой группы надёжны, последовательны, хорошо организованы. Они ценят трудолюбие, социальную ответственность, ведут себя сдержанно. Логистам можно доверить любую работу.'
ISTP_Gaben = 'мастер, виртуоз, логичный прагматик. Прямолинейный, последовательный человек, лояльно относящийся к коллегам. Не придерживается правил, непредсказуемый: может быть логичным, а может быть спонтанным. Ему необходимо живое общение, жесткий контроль.'


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
        self.root.geometry("1000x600")  # Увеличим размер окна для удобства

        self.embeddings = load_embeddings(embeddings_file)

        self.mode_var = StringVar(value="single")

        # Режим выбора
        mode_frame = Label(root)
        mode_frame.grid(row=0, column=0, columnspan=2, pady=10, sticky="w")  # Размещаем вверху

        Label(mode_frame, text="Режим:").pack(side='left')
        Radiobutton(mode_frame, text="Загрузить изображение", variable=self.mode_var, value="single").pack(side='left')
        Radiobutton(mode_frame, text="Выбрать папку", variable=self.mode_var, value="folder").pack(side='left')

        # Кнопка выбора
        self.select_button = Button(root, text="Выбор изображения/папки", command=self.select_input)
        self.select_button.grid(row=1, column=0, columnspan=2, pady=10, sticky="w")  # Размещаем под режимом

        # Кнопка "Описание типов"
        self.description_button = Button(root, text="Описание типов", command=self.show_type_descriptions)
        self.description_button.grid(row=1, column=1, pady=10, sticky="e")  # Размещаем справа

        # Отображение изображения (левый верхний угол)
        self.image_label = Label(root)
        self.image_label.grid(row=2, column=0, padx=20, pady=20, sticky="nw")  # Левый верхний угол с отступами

        # Описание под фото
        self.description_label = Label(root, text="", wraplength=300, justify="left")
        self.description_label.grid(row=3, column=0, padx=20, pady=10, sticky="w")  # Под фото

        # Результаты (правая часть)
        self.results_text = Text(root, wrap='word', height=20, width=50)
        self.results_text.grid(row=2, column=1, rowspan=2, padx=20, pady=20, sticky="nsew")  # Правая часть

        # Настройка растягивания столбцов и строк
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=2)
        root.grid_rowconfigure(2, weight=1)

    def select_input(self):
        if self.mode_var.get() == "single":
            file_path = filedialog.askopenfilename()
            if file_path:
                print(f"Selected image: {file_path}")
                self.classify_single_image(file_path)
            else:
                print("No image selected.")
                messagebox.showinfo("Info", "Не выбрано изображение.")
        else:
            folder_path = filedialog.askdirectory(title="Select Folder")
            if folder_path:
                print(f"Selected folder: {folder_path}")
                self.classify_folder(folder_path)

    def classify_single_image(self, file_path):
        self.display_image(file_path)
        result = classify_image(self.embeddings, file_path)

        if result:
            best_group, scores = result
            description = self.get_description(best_group)
            self.description_label.config(text=f"Наиболее вероятный тип: {best_group}\n\nОписание: {description}")

            # Сортируем результаты по расстоянию (RMSE)
            sorted_scores = sorted(scores.items(), key=lambda x: x[1])

            # Выводим отсортированные результаты в правую часть
            self.results_text.delete(1.0, 'end')
            self.results_text.insert('end', "Расстояния:\n" +
                                    "\n".join(f"{group}: {score:.2f}" for group, score in sorted_scores))
        else:
            messagebox.showerror("Error", "Не удалось классифицировать изображение.")

    def classify_folder(self, folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('png', 'jpg', 'jpeg'))]

        if not image_files:
            messagebox.showerror("Error", "В папке нет изображений.")
            return

        first_image_path = os.path.join(folder_path, image_files[0])
        self.display_image(first_image_path)

        most_common_class, classifications = process_images(self.embeddings, folder_path)
        if classifications:
            # Сортируем результаты по количеству вхождений
            sorted_classifications = sorted(Counter(classifications).items(), key=lambda x: x[1], reverse=True)

            # Выводим описание наиболее вероятного типа
            description = self.get_description(most_common_class)
            self.description_label.config(text=f"Наиболее вероятный тип: {most_common_class}\n\nОписание: {description}")

            # Выводим отсортированные результаты в правую часть
            self.results_text.delete(1.0, 'end')
            self.results_text.insert('end', f"Наиболее вероятный тип: {most_common_class}\n\n"
                                           "Результаты классификации:\n" +
                                    "\n".join(f"{idx + 1}. {cls} (количество: {count})" for idx, (cls, count) in enumerate(sorted_classifications)))
            print(f"Folder classification result: {most_common_class}")
        else:
            messagebox.showerror("Error", "Не удалось классифицировать ни одно изображение.")
            print("No images classified.")

    def display_image(self, file_path):
        try:
            image = Image.open(file_path)
            if image.mode == 'RGBA':
                image = image.convert('RGB')
            image.thumbnail((300, 300))  # Уменьшаем изображение для отображения
            self.tk_image = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.tk_image)
            self.image_label.image = self.tk_image
            print("Image displayed successfully.")
        except Exception as e:
            print(f"Error displaying image: {e}")
            messagebox.showerror("Error", "Не удалось отобразить изображение.")

    def get_description(self, best_group):
        descriptions = {
            "ENFJ_Gamlet": ENFJ_Gamlet,
            "ENFP_Geksli": ENFP_Geksli,
            "ENTJ_Djeck_London": ENTJ_Djeck_London,
            "ENTP_Don_kihot": ENTP_Don_kihot,
            "ESFJ_Gygo": ESFJ_Gygo,
            "ESFP_Napoleon": ESFP_Napoleon,
            "ESTJ_Shteerlic": ESTJ_Shteerlic,
            "ESTP_Jukov": ESTP_Jukov,
            "INFJ_Dostoevsky": INFJ_Dostoevsky,
            "INFP_Esenin": INFP_Esenin,
            "INTJ_Robesper": INTJ_Robesper,
            "INTP_Balzack": INTP_Balzack,
            "ISFJ_Dryzer": ISFJ_Dryzer,
            "ISFP_Dyma": ISFP_Dyma,
            "ISTJ_Gorkiy": ISTJ_Gorkiy,
            "ISTP_Gaben": ISTP_Gaben,
        }
        return descriptions.get(best_group, "Описание отсутствует.")

    def show_type_descriptions(self):
        # Создаем новое окно для отображения описаний
        descriptions_window = Toplevel(self.root)
        descriptions_window.title("Описание всех типов")
        descriptions_window.geometry("600x400")

        # Создаем текстовое поле для отображения описаний
        descriptions_text = Text(descriptions_window, wrap='word', height=25, width=70)
        descriptions_text.pack(padx=10, pady=10, fill='both', expand=True)

        # Добавляем описания всех типов
        descriptions = {
            "ENFJ_Gamlet": ENFJ_Gamlet,
            "ENFP_Geksli": ENFP_Geksli,
            "ENTJ_Djeck_London": ENTJ_Djeck_London,
            "ENTP_Don_kihot": ENTP_Don_kihot,
            "ESFJ_Gygo": ESFJ_Gygo,
            "ESFP_Napoleon": ESFP_Napoleon,
            "ESTJ_Shteerlic": ESTJ_Shteerlic,
            "ESTP_Jukov": ESTP_Jukov,
            "INFJ_Dostoevsky": INFJ_Dostoevsky,
            "INFP_Esenin": INFP_Esenin,
            "INTJ_Robesper": INTJ_Robesper,
            "INTP_Balzack": INTP_Balzack,
            "ISFJ_Dryzer": ISFJ_Dryzer,
            "ISFP_Dyma": ISFP_Dyma,
            "ISTJ_Gorkiy": ISTJ_Gorkiy,
            "ISTP_Gaben": ISTP_Gaben,
        }

        # Формируем текст для отображения
        descriptions_text.insert('end', "Описание всех типов:\n\n")
        for type_name, description in descriptions.items():
            descriptions_text.insert('end', f"{type_name}:\n{description}\n\n")

        # Делаем текстовое поле доступным только для чтения
        descriptions_text.config(state='disabled')


# === Запуск приложения ===
if __name__ == "__main__":
    try:
        root = Tk()

        app = ImageClassifierApp(root)

        root.mainloop()

    except Exception as e:
        print(f"Application error: {e}")
