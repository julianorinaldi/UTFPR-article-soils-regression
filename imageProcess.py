# -*- coding: utf-8 -*-

import tensorflow as tf  # Trabalhar com aprendizado de máquinas
from modelSet import ModelSet

from entityModelConfig import ModelConfig

# Carregamento de imagem para objeto e tratamento de imagem
def image_processing(modelConfig : ModelConfig, imageFilePath : str):
    img_path = f'{modelConfig.dirBaseImg}/{imageFilePath}'
    image = tf.keras.preprocessing.image.load_img(
        img_path, target_size=(modelConfig.imageDimensionX, modelConfig.imageDimensionY))
    image = tf.keras.preprocessing.image.img_to_array(image)
    if (modelConfig.argsPreprocess):
        # https://keras.io/api/applications/resnet/#resnet50-function
        # Note: each Keras Application expects a specific kind of input preprocessing.
        # For ResNet, call tf.keras.applications.resnet.preprocess_input on your inputs before passing them to the model.
        # resnet.preprocess_input will convert the input images from RGB to BGR, then will zero-center each color channel with respect to the ImageNet dataset, without scaling.
        
        if modelConfig.modelSet == ModelSet.ResNet50:
            image = tf.keras.applications.resnet50.preprocess_input(image)
        elif modelConfig.modelSet == ModelSet.ResNet101 or modelConfig.modelSet == ModelSet.ResNet152:
            image = tf.keras.applications.resnet.preprocess_input(image)

    return image
