import csv
import numpy as np
from deepface import DeepFace
from collections import defaultdict
import os
from collections import Counter

def load_embeddings(csv_file):
    """
    Загружает векторы из CSV файла и группирует их по типам (папкам).
    """
    embeddings = defaultdict(list)
    with open(csv_file, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Пропускаем заголовок
        for row in reader:
            file_path = row[0]
            group = file_path.split(os.sep)[-2]  # Имя папки
            vector = np.array(row[1:], dtype=float)
            embeddings[group].append(vector)
    return embeddings


def extract_embedding(image_path):
    """
    Извлекает векторное представление входного изображения.
    """
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name='Facenet', enforce_detection=False)
        return np.array(embedding[0]['embedding']) if embedding else None
    except Exception as e:
        print(f"Error processing input image: {e}")
        return None


def calculate_rmse(vector, group_vectors):
    """
    Вычисляет RMSE между входным вектором и каждым вектором группы.
    """
    diffs = [np.linalg.norm(vector - gv) for gv in group_vectors]
    return np.mean(diffs)


def classify_image(embeddings, input_image_path):
    """
    Классифицирует входное изображение по типу на основе минимального RMSE.
    """
    input_vector = extract_embedding(input_image_path)
    if input_vector is None:
        print("Failed to extract embedding from the input image.")
        return None

    scores = {}
    for group, group_vectors in embeddings.items():
        rmse = calculate_rmse(input_vector, group_vectors)
        scores[group] = rmse

    # Найти группу с минимальным RMSE
    best_group = min(scores, key=scores.get)
    return best_group, scores


if __name__ == "__main__":

    # Путь к CSV файлу с векторами
    csv_file = "embeddings.csv"  # Замените на путь к вашему файлу

    # Путь к папке с изображениями, которые нужно классифицировать
    input_folder = "test_images/Kianu"  # Укажите путь к папке с входными изображениями
    # input_folder = None
    input_image_path = "test_images/DiCaprio/img_2.png"  # Укажите путь к входному изображению


    # Загрузка векторов из CSV
    embeddings = load_embeddings(csv_file)

    if input_folder:
        # Словарь для хранения результатов классификации
        classification_results = []

        # Обработка всех изображений в папке
        for file_name in os.listdir(input_folder):
            if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                input_image_path = os.path.join(input_folder, file_name)
                print(f"Processing {input_image_path}...")

                # Классификация входного изображения
                result = classify_image(embeddings, input_image_path)
                if result:
                    best_group, scores = result
                    classification_results.append(best_group)
                    print(f"Image: {file_name} -> Type: {best_group}")
                else:
                    print(f"Failed to classify image: {file_name}")

        # Определение наиболее часто встречающегося класса
        if classification_results:
            most_common_class, count = Counter(classification_results).most_common(1)[0]
            print("\n--- Final Result ---")
            print(f"The most common type is: {most_common_class}")
            print(f"Occurrences: {count} out of {len(classification_results)}")
        else:
            print("No classifications were successful.")

    else:

        # Классификация входного изображения
        result = classify_image(embeddings, input_image_path)

        if result:
            best_group, scores = result
            print(f"The input image belongs to the type: {best_group}")
            print("Scores by group:")
            for group, score in scores.items():
                print(f"{group}: {score}")
