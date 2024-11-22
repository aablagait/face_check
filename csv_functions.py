'''Здесь описаны функции при работе с .csv файлом.'''

import numpy as np
import pandas as pd
from deepface import DeepFace
from scipy.spatial.distance import pdist
import numpy as np
from scipy.spatial.distance import cosine

from constants import backends, models, alignment_modes


def get_vector(img_path):
    '''Получение векторного представления.
    Это вспомогательная функция, используется внтури других'''
    embedding_objs = DeepFace.represent(
        img_path=img_path,
        model_name=models[0],
    )
    vector = embedding_objs[0]['embedding']
    return vector


def add_vector(df):
    '''Функция нужна только один раз, когда в .csv файл
    добавляется столбец "vector" который содержит
    векторное представление изображений'''
    df['vector'] = df['image'].apply(lambda img_path: get_vector(img_path))


def search_user(df, img_path):
    '''Поиск самого похожего пользователя.
    Возвращает пару значений - строку в датафрейме с
    данными этого пользователя и косинусное расстояние
    между ним и искомой фотографией'''
    vector = get_vector(img_path)

    distances = df['vector'].apply(lambda x: cosine(x, vector))
    twin = distances.idxmin()
    min_distance = distances[twin]

    return df.loc[twin], min_distance


def distance_to_percents(distance):
    return f"{(1 - np.round(distance, decimals=3)) * 100:.1f}%"



# Пример работы с .csv файлом
# df = pd.read_csv('man.csv', delimiter=',')
# searcher = 'images/img_11.png' # фотография двойника, которого надо найти
#
# add_vector(df) # создание столбца 'vector', использовать только при ервом включении
# twin, distance = search_user(df, searcher) # поиск двойника
#
#
# print(df.head()) # первые 10 строк датафрейма, для примера
# print("Наиболее похожий тип:", twin)
# print('Он похож на ', distance_to_percents(distance)) # схожесть в процентах, если больше 40%, то это один и тот же человек