# -*- coding: utf-8 -*-

import argparse
import numpy as np # Trabalhar com array
import pandas as pd # Trabalhar com análise de dados, importação, etc.
import tensorflow as tf # Trabalhar com aprendizado de máquinas
import keras # Trabalhar com aprendizado de máquinas
import cv2 # Trabalhar com processamento de imagens
import os
import random
from matplotlib import pyplot as plt # Matplotlib Plot
from tqdm import tqdm # Facilita visualmente a iteração usado no "for"
from sklearn.metrics import r2_score # Avaliação das Métricas
from coreProcess import image_processing

#3 = INFO, WARNING, and ERROR messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", help="Para listar os prints de Debug: True para Sim, False para Não (default)") 
parser.add_argument("-n", "--name", help="Nome do arquivo do modelo .h5")
parser.add_argument("-p", "--preprocess", help="Preprocessar imagem: True para Sim (default), False para Não") 

args = parser.parse_args()

if not (args.name):
    print("Há parâmetros faltantes. Utilize -h ou --help para ajuda!")
    exit(1)

if (args.preprocess) and (args.preprocess != "True") and (args.preprocess != "False"):
    print("Preprocessar imagem: True para Sim, False para Não")
    exit(1)
else:
    preprocess = True if args.preprocess is None else eval(args.preprocess)
    
if (args.debug) and (args.debug != "True") and (args.debug != "False"):
    print("Para listar os prints de Debug: True para Sim, False para Não (default)")
    exit(1)
else:
    debug = False if args.debug is None else eval(args.debug)
  
if (debug):  
    print(f'Versão do tensorflow: {tf.__version__}')
    print(f'Eager: {tf.executing_eagerly()}')
    print(f"GPU: {tf.config.list_physical_devices('GPU')}")
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

physical_devices = tf.config.list_physical_devices('GPU')

if len(physical_devices) == 2:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
if (debug):
    print('Number of devices =====>: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():
        
    # Carregamento do Dataset
    df_test = pd.read_csv('dataset/csv/Dataset256x256-Teste.csv')

    df_test = df_test.drop(columns={"class", "qtd_mat_org", "nitrog_calc", "amostra", "classe", "tamanho"})

    # Definindo o tamanho das imagens
    imageDimensionX = 256
    imageDimensionY = 256

    # Path Dir Teste
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
    
    for imageFilePath in tqdm(test_imagefiles.tolist()[:qtd_imagens]):
        image_list_test.append(image_processing(dir_name_test, imageFilePath, imageDimensionX, imageDimensionY, preprocess))
        
    # Transformando em array a lista de imagens (Test)
    X_test = np.array(image_list_test)
    if (debug):
        print(f'Shape X_test: {X_test.shape}')

    Y_test_carbono = np.array(df_test['teor_carbono'].tolist()[:qtd_imagens])
    if (debug):
        print(f'Shape Y_test_carbono: {Y_test_carbono.shape}')

    # Carregando Modelo
    resnet_model = tf.keras.models.load_model(args.name)
    #print(resnet_model.summary())

    # Trabalhando com R2
    prediction = resnet_model.predict(X_test)

    r2 = r2_score(Y_test_carbono, prediction)
    print(f'====================================================')
    print(f'====================================================')
    print(f'=========>>>>> R2: {r2} <<<<<=========')
    print(f'====================================================')
    print(f'====================================================')
    
    for i in [0,10,100,500,1000,2500,3500]:
        indexImg = random.randint(i, len(image_list_test))
        img_path = f'{dir_name_test}/{test_imagefiles[indexImg]}'
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256, 3))
        x = tf.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.keras.applications.resnet50.preprocess_input(x)

        #x2 = cv2.imread(f'{dir_name_test}/{test_imagefiles[index]}')
        #x2 = np.expand_dims(x2, axis=0)

        ResNet50 = resnet_model.predict(x)
        #CV2 = resnet_model.predict(x2)
        Real = df_test.teor_carbono[indexImg]

        #print(f'CV2: {CV2.item(0)} => Diferença: {Real - CV2.item(0)}')
        print(f'Image[{indexImg}]: {test_imagefiles[indexImg]} => {df_test.teor_carbono[indexImg]}')
        print(f'ResNet50[{indexImg}]: {ResNet50.item(0)} => Diferença: {Real - ResNet50.item(0)}')
        print("")

