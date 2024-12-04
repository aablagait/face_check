import numpy as np
import pandas as pd
from deepface import DeepFace
from scipy.spatial.distance import pdist
import numpy as np
from scipy.spatial.distance import cosine

from constants import backends, models, alignment_modes, users
import csv_functions as cf


df = pd.read_csv('art.csv', delimiter=',') # загружаем .csv файл в переменную df
searcher = 'images/pas.jpg' # фотография двойника, которого надо найти

cf.add_vector(df) # создание столбца 'vector', использовать только при первом включении
twin, distance = cf.search_user(df, searcher) # поиск двойника


# print(df.head()) # первые 10 строк датафрейма, для примера
print("Наиболее похожий тип:", twin)
print('Он похож на ', cf.distance_to_percents(distance)) # схожесть в процентах, если больше 40%, то это один и тот же человек
