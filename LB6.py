from tensorflow.keras.models import load_model
# from google.colab
import files
import tkinter as tk
from tkinter import filedialog, messagebox
from IPython.display import Image
from tensorflow.keras.preprocessing import image
import numpy as np

classes = ['футболка', 'брюки', 'свитер', 'платье', 'пальто', 'туфли', 'рубашка', 'кроссовки', 'сумка', 'ботинки']
model = load_model('fashion_mnist_dense.h5')

model.summary()

"""## Загружаем в Colab изображение для распознавания"""

# f = files.upload()


img_path = 'загруженное.jfif'

Image(img_path, width=150, height=150)

"""## Распознаем изображение

Загружаем изображение из файла с помощью инструментов Keras
"""

img = image.load_img(img_path, target_size=(28, 28), color_mode = "grayscale")

"""Предварительная обработка изображения"""

# Преобразуем картинку в массив
x = image.img_to_array(img)
# Меняем форму массива в плоский вектор
x = x.reshape(1, 784)
# Инвертируем изображение
x = 255 - x
# Нормализуем изображение
x /= 255

"""Запускаем распознавание"""

prediction = model.predict(x)

"""Результаты распознавания"""

prediction

prediction = np.argmax(prediction)
print("Номер класса:", prediction)
print("Название класса:", classes[prediction])