# -*- coding: utf-8 -*-

import argparse
import numpy as np # Trabalhar com array
import pandas as pd # Trabalhar com análise de dados, importação, etc.
from matplotlib import pyplot as plt # Matplotlib Plot
from tqdm import tqdm # Facilita visualmente a iteração usado no "for"
import tensorflow as tf # Trabalhar com aprendizado de máquinas
import os
from coreProcess import image_processing

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", help="Nome do arquivo de saída do modelo .h5")

args = parser.parse_args()

if not (args.name):
    print("Há parâmetros faltantes. Utilize -h ou --help para ajuda!")
    exit(1)
    
print(f'Versão do tensorflow: {tf.__version__}')
print(f'Eager: {tf.executing_eagerly()}')
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

physical_devices = tf.config.list_physical_devices('GPU')

if len(physical_devices) == 2:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

strategy = tf.distribute.MirroredStrategy(cross_device_ops = tf.distribute.HierarchicalCopyAllReduce())
print('Number of devices =====>: {}'.format(strategy.num_replicas_in_sync))

with strategy.scope():

    # Carregamento do Dataset
    df_train = pd.read_csv('dataset/csv/Dataset256x256-Treino.csv')

    df_train = df_train.drop(columns={"class", "qtd_mat_org", "nitrog_calc", "amostra", "classe", "tamanho"})
    df_train = df_train.sample(frac=1, random_state=1, ignore_index=True)
    
    # Definindo o tamanho das imagens
    imageDimensionX = 256
    imageDimensionY = 256

    # Path Dir Treino
    dir_name_train = "dataset/images/treinamento-solo-256x256"

    # Separando apenas nomes dos arquivos
    train_imagefiles = df_train["arquivo"]

    # Removendo coluna arquivo para normalização
    df_train = df_train.drop(columns={"arquivo"})

    # Quantidade de imagens usadas para a rede.
    qtd_imagens = 10000
    qtd_canal_color = 3

    # Normalização Dataset Treinamento
    train_stats = df_train.describe()
    train_stats = train_stats.transpose()
    df_train = (df_train - train_stats['mean']) / train_stats['std']

    # Array com as imagens a serem carregadas de treino
    image_list_train = []

    for imageFilePath in tqdm(train_imagefiles.tolist()[:qtd_imagens]):
        image_list_train.append(image_processing(dir_name_train, imageFilePath, imageDimensionX, imageDimensionY, False))

    # Transformando em array a lista de imagens (Treino)
    X_train =  np.array(image_list_train)
    print(f'Shape X_train: {X_train.shape}')

    Y_train_carbono = np.array(df_train['teor_carbono'].tolist()[:qtd_imagens])
    print(f'Shape Y_train_carbono: {Y_train_carbono.shape}')

    resnet_model = tf.keras.models.Sequential()

    pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(imageDimensionX, imageDimensionY, qtd_canal_color),
                   pooling='avg', classes=1,
                   weights='imagenet')
    for layer in pretrained_model.layers:
            layer.trainable=True

    resnet_model.add(pretrained_model)
    resnet_model.add(tf.keras.layers.Flatten())
    resnet_model.add(tf.keras.layers.Dense(512, activation='relu'))
    resnet_model.add(tf.keras.layers.Dense(256, activation='relu'))
    resnet_model.add(tf.keras.layers.Dropout(0.5))
    resnet_model.add(tf.keras.layers.Dense(1))

    print(resnet_model.summary())

    opt = tf.keras.optimizers.adam(0.0001)
    resnet_model.compile(optimizer=opt,loss='mse',metrics=['mae', 'mse'])
    history = resnet_model.fit(X_train, Y_train_carbono, validation_split=0.3, epochs=100, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

    resnet_model.save(args.name)

