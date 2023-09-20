# -*- coding: utf-8 -*-

import argparse
import os
import random

import cv2  # Trabalhar com processamento de imagens
import keras  # Trabalhar com aprendizado de máquinas
import numpy as np  # Trabalhar com array
import pandas as pd  # Trabalhar com análise de dados, importação, etc.
import tensorflow as tf  # Trabalhar com aprendizado de máquinas
from matplotlib import pyplot as plt  # Matplotlib Plot
from sklearn.metrics import r2_score  # Avaliação das Métricas
from tqdm import tqdm  # Facilita visualmente a iteração usado no "for"

from coreProcess import image_processing

prefix = ">>>>>>>>>>>>>>>>>"

# 3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true",
                    help="Para listar os prints de Debug")
parser.add_argument(
    "-n", "--name", help="Nome do arquivo de saída do modelo .h5")
parser.add_argument("-p", "--preprocess", action="store_true",
                    help="Preprocessar imagem 'resnet50.preprocess_input(...)'")
parser.add_argument(
    "-g", "--gpu", help="Index da GPU a process. Considere 0 a primeira. Caso use mais uma, ex. para duas: 0,1")

args = parser.parse_args()

if not (args.name):
    print(f'{prefix} Há parâmetros faltantes. Utilize -h ou --help para ajuda!')
    exit(1)

physical_devices = tf.config.list_physical_devices('GPU')
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if not (args.gpu):
    if (len(physical_devices) > 0):
        #os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        #tf.config.set_visible_devices(physical_devices[1], 'GPU')
        #tf.config.experimental.set_memory_growth(physical_devices[1], True)
        tf.config.set_visible_devices(physical_devices[1], 'GPU')
        tf.config.experimental.set_memory_growth(physical_devices[1], True)        
        
else:
    gpusArray = args.gpu.split(',')
    gpu_count = len(gpusArray)
    gpu = ",".join(str(int(g) + 1) for g in gpusArray)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu

# Infos da GPU e Framework
if (args.debug):
    print(f'{prefix} Tensorflow Version: {tf.__version__}')
    print(f'{prefix} Amount of GPU Available: {physical_devices}')
    print(f'{prefix} Indexes of selected GPUs: {os.environ["CUDA_VISIBLE_DEVICES"]}')

# Estratégia para trabalhar com Multi-GPU
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with strategy.scope():

    # Carregamento do Dataset
    df_test = pd.read_csv('dataset/csv/Dataset256x256-Teste.csv')

    # Removendo colunas desnecessárias
    df_test = df_test.drop(
        columns={"class", "qtd_mat_org", "nitrog_calc", "amostra", "classe", "tamanho"})

    # Definindo o tamanho das imagens
    imageDimensionX = 256
    imageDimensionY = 256

    # # Path Diretório Teste das Imagens
    dir_name_test = "dataset/images/teste-solo-256x256"

    # Separando apenas nomes dos arquivos
    test_imagefiles = df_test["arquivo"]

    # Removendo coluna arquivo para normalização
    df_test = df_test.drop(columns={"arquivo"})

    # Quantidade de imagens usadas para a rede.
    qtd_imagens = 10000
    qtd_canal_color = 3

    # Normalização Dataset Teste
    test_stats = df_test.describe()
    test_stats = test_stats.transpose()
    df_test = (df_test - test_stats['mean']) / test_stats['std']

    # Array com as imagens a serem carregadas de treino
    image_list_test = []

    if (args.debug):
        print(f'{prefix} Preprocess: {args.preprocess}')
    for imageFilePath in tqdm(test_imagefiles.tolist()[:qtd_imagens]):
        # Carregamento de imagens Com/Sem Preprocessamento (args.preprocess)
        image_list_test.append(image_processing(
            dir_name_test, imageFilePath, imageDimensionX, imageDimensionY, args.preprocess))

    # Transformando em array a lista de imagens (Test)
    X_test = np.array(image_list_test)
    if (args.debug):
        print(f'{prefix} Shape X_test: {X_test.shape}')

    Y_test_carbono = np.array(df_test['teor_carbono'].tolist()[:qtd_imagens])
    if (args.debug):
        print(f'{prefix} Shape Y_test_carbono: {Y_test_carbono.shape}')

    # Carregando Modelo
    resnet_model = tf.keras.models.load_model(args.name)
    if (args.debug):
        print(resnet_model.summary())

    # Fazendo a predição sobre os dados de teste
    prediction = resnet_model.predict(X_test)

    # Avaliando com R2
    r2 = r2_score(Y_test_carbono, prediction)
    print(f'====================================================')
    print(f'====================================================')
    print(f'=========>>>>> R2: {r2} <<<<<=========')
    print(f'====================================================')
    print(f'====================================================')

    # Trazendo algumas amostras aleatórias ...
    for i in [0, 10, 50, 60, 100, 200, 300, 400, 500, 1000, 2000, 3000, 3500]:
        # Essa linha abaixo garante aleatoriedade
        indexImg = random.randint(i, len(image_list_test))
        img_path = f'{dir_name_test}/{test_imagefiles[indexImg]}'
        img = tf.keras.preprocessing.image.load_img(
            img_path, target_size=(256, 256, 3))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.resnet50.preprocess_input(x)

        # x2 = cv2.imread(f'{dir_name_test}/{test_imagefiles[index]}')
        # x2 = np.expand_dims(x2, axis=0)

        ResNet50 = resnet_model.predict(x)
        # CV2 = resnet_model.predict(x2)
        Real = df_test.teor_carbono[indexImg]

        # print(f'CV2: {CV2.item(0)} => Diferença: {Real - CV2.item(0)}')
        print(
            f'{prefix} Image[{indexImg}]: {test_imagefiles[indexImg]} => {df_test.teor_carbono[indexImg]}')
        print(
            f'{prefix} ResNet50[{indexImg}]: {ResNet50.item(0)} => Diferença: {Real - ResNet50.item(0)}')
        print("")
