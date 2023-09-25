# -*- coding: utf-8 -*-

import argparse
import os
import random

import numpy as np  # Trabalhar com array
import tensorflow as tf  # Trabalhar com aprendizado de máquinas
from sklearn.metrics import r2_score  # Avaliação das Métricas
from tqdm import tqdm  # Facilita visualmente a iteração usado no "for"

from imageProcess import image_load, image_convert_array, image_processing
from datasetProcess import dataset_process
from modelSet import ModelSet
from entityModelConfig import ModelConfig
from modelTransferLearningProcess import modelTransferLearningProcess

prefix = ">>>>>>>>>>>>>>>>>"

# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true", help="Para listar os prints de Debug")
parser.add_argument("-n", "--name", help="Nome do arquivo de saída do modelo .h5")
parser.add_argument("-p", "--preprocess", action="store_true", help="Preprocessar imagem 'resnet50.preprocess_input(...)'")

args = parser.parse_args()

if not (args.name):
    print(f'{prefix} Há parâmetros faltantes. Utilize -h ou --help para ajuda!')
    exit(1)

physical_devices = tf.config.list_physical_devices('GPU')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Infos da GPU e Framework
if (args.debug):
    print(f'{prefix} Tensorflow Version: {tf.__version__}')
    print(f'{prefix} Amount of GPU Available: {physical_devices}')

# Definindo Modelo de TransferLearning e Configurações
modelSet = ModelSet.ResNet152
imageDimensionX = 256
imageDimensionY = 256
qtd_canal_color = 3
dir_base_img = 'dataset/images/teste-solo-256x256'
pathCsv = 'dataset/csv/Dataset256x256-Teste.csv'
modelConfig = ModelConfig(modelSet, pathCsv, dir_base_img, imageDimensionX, imageDimensionY, qtd_canal_color,
                          args.name, args.debug, False, args.preprocess)

# Estratégia para trabalhar com Multi-GPU
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with strategy.scope():

    df, imageNamesList = dataset_process(modelConfig)

    # Quantidade de imagens usadas para a rede.
    qtd_imagens = len(df)
    if (modelConfig.argsDebug):
        print(f'{prefix} Preprocess: {args.preprocess}')
    
    # Array com as imagens a serem carregadas de treino
    imageArray = image_load(modelConfig, imageNamesList, qtd_imagens)

    X_, Y_carbono = image_convert_array(modelConfig, imageArray, df, qtd_imagens)

    # Carregando Modelo
    resnet_model = tf.keras.models.load_model(modelConfig.argsNameModel)
    if (modelConfig.argsDebug):
        print(f'{prefix}')
        print(resnet_model.summary())
        print(f'{prefix}')

    # Trazendo algumas amostras aleatórias ...
    for i in [0, 10, 50, 60, 100, 200, 300, 400, 500, 1000, 2000, 3000, 3500]:
        # Essa linha abaixo garante aleatoriedade
        indexImg = random.randint(i, len(imageNamesList))
        img_path = f'{imageNamesList[indexImg]}'
        img = image_processing(modelConfig, img_path)

        predictValue = resnet_model.predict(img)
        Real = df.teor_carbono[indexImg]

        print(f'{prefix} Original image[{indexImg}]: {imageNamesList[indexImg]} => {df.teor_carbono[indexImg]}')
        print(f'{prefix} {modelConfig.modelSet.name}[{indexImg}]: {predictValue.item(0)} => Diff: {Real - predictValue.item(0)}')
        print("")

    # Fazendo a predição sobre os dados de teste
    prediction = resnet_model.predict(X_)

    # Avaliando com R2
    r2 = r2_score(Y_carbono, prediction)
    print()
    print(f'====================================================')
    print(f'====================================================')
    print(f'=========>>>>> R2: {r2} <<<<<=========')
    print(f'====================================================')
    print(f'====================================================')

    print()
    print(f"{prefix} Info parameters: ")
    print(f"{prefix}{prefix} -d (--debug): {args.debug}")
    print(f"{prefix}{prefix} -n (--name): {args.name}")
    print(f"{prefix}{prefix} -p (--preprocess): {args.preprocess}")