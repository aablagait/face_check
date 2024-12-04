import os
import csv

import numpy as np
from deepface import DeepFace
from PIL import Image


def extract_embeddings(image_path):
    try:
        # DeepFace используем для извлечения эмбеддингов
        embedding = DeepFace.represent(img_path=image_path, model_name='Facenet', enforce_detection=False)
        return embedding[0]['embedding'] if embedding else None
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None


def process_images_in_folders(base_folder, output_csv):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["File Path"] + [f"Feature_{i}" for i in range(128)])  # 128 - длина вектора в Facenet

        for root, dirs, files in os.walk(base_folder):
            for file_name in files:
                if file_name.lower().endswith(('png', 'jpg', 'jpeg')):
                    image_path = os.path.join(root, file_name)
                    print(f"Processing {image_path}...")

                    # Извлечение векторов
                    embedding = extract_embeddings(image_path)
                    if embedding:
                        writer.writerow([image_path] + embedding)
                    else:
                        print(f"Skipping {image_path} due to errors.")


if __name__ == "__main__":
    base_folder = "types"  # Замените на путь к вашей папке с фотографиями
    output_csv = "embeddings.csv"
    process_images_in_folders(base_folder, output_csv)
