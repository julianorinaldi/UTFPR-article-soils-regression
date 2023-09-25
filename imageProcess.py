# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf  # Trabalhar com aprendizado de máquinas
import numpy as np
from tqdm import tqdm  # Facilita visualmente a iteração usado no "for"

from modelSet import ModelSet
from entityModelConfig import ModelConfig

def image_load(modelConfig : ModelConfig, imagefiles : list, qtd_imagens : int):
        # Array com as imagens a serem carregadas de treino
    imageArray = []    
    for imageFilePath in tqdm(imagefiles[:qtd_imagens]):
        imageArray.append(image_processing(modelConfig, imageFilePath))
    
    return imageArray
        
# Carregamento de imagem para objeto e tratamento de imagem
def image_processing(modelConfig : ModelConfig, imageName : str):
    print(f'modelConfig: {modelConfig.dirBaseImg}')
    img_path = f'{modelConfig.dirBaseImg}/{imageName}'
    print(f'Imagem: {img_path}')
    image = tf.keras.preprocessing.image.load_img(
        path = img_path, target_size=(modelConfig.imageDimensionX, modelConfig.imageDimensionY))
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

def image_convert_array(modelConfig : ModelConfig, imagesArray : list, df : pd.DataFrame, qtd_imagens : int):
    # Transformando em array a lista de imagens (Treino)
    X_ = np.array(imagesArray)
    if (modelConfig.argsDebug):
        print(f'{modelConfig.printPrefix} Shape X_: {X_.shape}')

    # *******************************************************
    # Neste momento apenas trabalhando com valores de Carbono
    # *******************************************************
    Y_carbono = np.array(df['teor_carbono'].tolist()[:qtd_imagens])
    if (modelConfig.argsDebug):
        print(f'{modelConfig.printPrefix} Shape Y_carbono: {Y_carbono.shape}')
        
    #Y_train_nitrogenio = np.array(df_train['teor_nitrogenio'].tolist()[:qtd_imagens])
    #print(f'Shape Y_train_nitrogenio: {Y_train_nitrogenio.shape}')
    
    return X_, Y_carbono