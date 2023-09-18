# -*- coding: utf-8 -*-

import tensorflow as tf # Trabalhar com aprendizado de máquinas

# Carregamento de imagem para objeto e tratamento de imagem
def image_processing(dir_name_base, imageFilePath, dimensionX, dimensionY, qtd_canal_color):
    img_path = f'{dir_name_base}/{imageFilePath}'
    image = tf.keras.preprocessing.image.load_img(img_path, target_size=(dimensionX, dimensionY))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.resnet50.preprocess_input(image)

    return image