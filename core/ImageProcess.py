# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf  # Trabalhar com aprendizado de máquinas
import numpy as np
from tqdm import tqdm  # Facilita visualmente a iteração usado no "for"

from core.ModelSetEnum import ModelSetEnum
from core.ModelConfig import ModelConfig


class ImageProcess:
    def __init__(self, config: ModelConfig):
        self.config = config

    def image_load(self, imagefiles: list, qtd_imagens: int):
        # Array com as imagens a serem carregadas de treino
        image_array = []
        for imageFilePath in tqdm(imagefiles[:qtd_imagens]):
            image_array.append(self.image_processing(imageFilePath))

        return image_array

    # Carregamento de imagem para objeto e tratamento de imagem
    def image_processing(self, image_name: str):
        img_path = f'{self.config.dirBaseImg}/{image_name}'
        image = tf.keras.preprocessing.image.load_img(
            path=img_path, target_size=(self.config.imageDimensionX, self.config.imageDimensionY))
        image = tf.keras.preprocessing.image.img_to_array(image)
        if self.config.argsPreprocess:
            # https://keras.io/api/applications Note: each Keras Application expects a specific kind of input
            # preprocessing. For ResNet, call tf.keras.applications.resnet.preprocess_input on your inputs before
            # passing them to the model. resnet.preprocess_input will convert the input images from RGB to BGR,
            # then will zero-center each color channel with respect to the ImageNet dataset, without scaling.

            if self.config.modelSetEnum == ModelSetEnum.ResNet50:
                image = tf.keras.applications.resnet50.preprocess_input(image)
            elif (self.config.modelSetEnum == ModelSetEnum.ResNet101
                  or self.config.modelSetEnum == ModelSetEnum.ResNet152):
                image = tf.keras.applications.resnet.preprocess_input(image)
            elif (self.config.modelSetEnum == ModelSetEnum.ConvNeXtBase
                  or self.config.modelSetEnum == ModelSetEnum.ConvNeXtXLarge):
                image = tf.keras.applications.convnext.preprocess_input(image)
            elif self.config.modelSetEnum == ModelSetEnum.EfficientNetB7:
                image = tf.keras.applications.efficientnet.preprocess_input(image)
            elif (self.config.modelSetEnum == ModelSetEnum.EfficientNetV2S
                  or self.config.modelSetEnum == ModelSetEnum.EfficientNetV2L):
                image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
            elif self.config.modelSetEnum == ModelSetEnum.InceptionResNetV2:
                image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)
            elif self.config.modelSetEnum == ModelSetEnum.DenseNet169:
                image = tf.keras.applications.densenet.preprocess_input(image)
            elif self.config.modelSetEnum == ModelSetEnum.VGG19:
                image = tf.keras.applications.vgg19.preprocess_input(image)
            else:
                error = "Modelo desconhecido"
                self.config.logger.log_info(f"Excetion: {error}")
                raise Exception(error)

        return image

    def image_convert_array(self, images_array: list, df: pd.DataFrame, qtd_imagens: int):
        # Transformando em array a lista de imagens (Treino)
        x_img_array = np.array(images_array)
        self.config.logger.log_debug(f"Shape X: {x_img_array.shape}")

        # *******************************************************
        # Neste momento apenas trabalhando com valores de Carbono
        # *******************************************************
        y_carbono = np.array(df['teor_carbono'].tolist()[:qtd_imagens])
        y_nitrogenio = np.array(df['teor_nitrogenio'].tolist()[:qtd_imagens])
        self.config.logger.log_debug(f"Shape Y: {y_carbono.shape}")
        self.config.logger.log_debug(f"Shape Y: {y_nitrogenio.shape}")

        return x_img_array, y_carbono, y_nitrogenio
