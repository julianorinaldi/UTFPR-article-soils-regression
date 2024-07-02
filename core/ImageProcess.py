# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf  # Trabalhar com aprendizado de máquinas
import numpy as np
from tqdm import tqdm  # Facilita visualmente a iteração usado no "for"

from dto.ModelSetEnum import ModelSetEnum
from dto.ConfigModelDTO import ConfigModelDTO


class ImageProcess:
    def __init__(self, config: ConfigModelDTO):
        self.config = config

    # Carregamento de imagem para objeto e tratamento de imagem
    def __image_processing(self, path_img: str, image_name: str) -> np.ndarray:
        img_path = f'{path_img}/{image_name}'
        image = tf.keras.preprocessing.image.load_img(
            path=img_path, target_size=(self.config.imageDimensionX, self.config.imageDimensionY))
        image = tf.keras.preprocessing.image.img_to_array(image)

        if self.config.argsPreprocess:
            preprocess_map = {
                ModelSetEnum.ResNet50: tf.keras.applications.resnet50.preprocess_input,
                ModelSetEnum.ResNet101: tf.keras.applications.resnet.preprocess_input,
                ModelSetEnum.ResNet152: tf.keras.applications.resnet.preprocess_input,
                ModelSetEnum.ConvNeXtBase: tf.keras.applications.convnext.preprocess_input,
                ModelSetEnum.ConvNeXtXLarge: tf.keras.applications.convnext.preprocess_input,
                ModelSetEnum.EfficientNetB7: tf.keras.applications.efficientnet.preprocess_input,
                ModelSetEnum.EfficientNetV2S: tf.keras.applications.efficientnet_v2.preprocess_input,
                ModelSetEnum.EfficientNetV2L: tf.keras.applications.efficientnet_v2.preprocess_input,
                ModelSetEnum.InceptionResNetV2: tf.keras.applications.inception_resnet_v2.preprocess_input,
                ModelSetEnum.DenseNet169: tf.keras.applications.densenet.preprocess_input,
                ModelSetEnum.VGG19: tf.keras.applications.vgg19.preprocess_input
            }

            preprocess_func = preprocess_map.get(self.config.modelSetEnum)
            if preprocess_func:
                image = preprocess_func(image)
            else:
                raise Exception("Modelo desconhecido para pré-processamento")

        return image

    # Carregar imagens
    def image_load(self, df: pd.DataFrame, path_img: str, qtd_imagens: int) -> list:
        if df["arquivo"] is None:
            raise Exception("O DataFrame deve conter a coluna 'arquivo'")

        image_files: list = df["arquivo"].to_list()

        # Array com as imagens a serem carregadas de treino
        image_array = []
        for img_file_path in tqdm(image_files[:qtd_imagens]):
            image_array.append(self.__image_processing(path_img, img_file_path))

        return image_array