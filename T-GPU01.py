# -*- coding: utf-8 -*-

# Imports
import argparse
import os

import numpy as np  # Trabalhar com array
import pandas as pd  # Trabalhar com análise de dados, importação, etc.
import tensorflow as tf  # Trabalhar com aprendizado de máquinas
from matplotlib import pyplot as plt  # Matplotlib Plot
from tqdm import tqdm  # Facilita visualmente a iteração usado no "for"

from coreProcess import image_processing

prefix = ">>>>>>>>>>>>>>>>>"

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true",
                    help="Para listar os prints de Debug")
parser.add_argument(
    "-n", "--name", help="Nome do arquivo de saída do modelo .h5")
parser.add_argument("-p", "--preprocess", action="store_true",
                    help="Preprocessar imagem 'resnet50.preprocess_input(...)'")
parser.add_argument(
    "-t", "--trainable", action="store_true", help="Define se terá as camadas do modelo de transfer-learning treináveis ou não")

args = parser.parse_args()

if not (args.name):
    print(f"{prefix} Há parâmetros faltantes. Utilize -h ou --help para ajuda!")
    exit(1)

physical_devices = tf.config.list_physical_devices('GPU')
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Infos da GPU e Framework
if (args.debug):
    print(f'{prefix} Tensorflow Version: {tf.__version__}')
    print(f'{prefix} Amount of GPU Available: {physical_devices}')
    #print(f'{prefix} Indexes of selected GPUs: {os.environ["CUDA_VISIBLE_DEVICES"]}')

# Estratégia para trabalhar com Multi-GPU
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with strategy.scope():

    # Carregamento do Dataset
    df_train = pd.read_csv('dataset/csv/Dataset256x256-Treino.csv')

    # Removendo colunas desnecessárias
    df_train = df_train.drop(
        columns={"class", "qtd_mat_org", "nitrog_calc", "amostra", "classe", "tamanho"})

    # *********************************
    # Excluindo Nitrogênio Por Enquanto
    # *********************************
    df_train = df_train.drop(columns={"teor_nitrogenio"})
    # *********************************

    # Randomizando
    df_train = df_train.sample(frac=1, random_state=1, ignore_index=True)

    # Definindo o tamanho das imagens
    imageDimensionX = 256
    imageDimensionY = 256

    # Path Diretório Treino das Imagens
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

    # Modelos disponíveis para Transfer-Learning
    # https://keras.io/api/applications/
    # ResNet50
    
    # include_top=False => Excluindo as camadas finais (top layers) da rede, que geralmente são usadas para classificação. Vamos adicionar nossas próprias camadas finais
    # input_shape=(imageDimensionX, imageDimensionY, qtd_canal_color) => Nosso array de imagens tem as dimensões (256, 256, 3)
    # pooling => Modo de pooling opcional para extração de recursos quando include_top for False [none, avg (default), max], passado por parâmetro, mas o default é avg.
    # classes=1 => Apenas uma classe de saída, no caso de regressão precisamos de valor predito para o carbono.
    # weights='imagenet' => Carregamento do modelo inicial com pesos do ImageNet, no qual no treinamento será re-adaptado.
    pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                      input_shape=(imageDimensionX, imageDimensionY, qtd_canal_color),
                                                      classes=1, weights='imagenet')


    # Array com as imagens a serem carregadas de treino
    image_list_train = []

    if (args.debug):
        print(f'{prefix} Preprocess: {args.preprocess}')
    for imageFilePath in tqdm(train_imagefiles.tolist()[:qtd_imagens]):
        # Carregamento de imagens Com/Sem Preprocessamento (args.preprocess)
        image_list_train.append(image_processing(pretrained_model, dir_name_train, imageFilePath, imageDimensionX, imageDimensionY, args.preprocess))

    # Transformando em array a lista de imagens (Treino)
    X_train = np.array(image_list_train)
    if (args.debug):
        print(f'{prefix} Shape X_train: {X_train.shape}')

    # *******************************************************
    # Neste momento apenas trabalhando com valores de Carbono
    # *******************************************************
    Y_train_carbono = np.array(df_train['teor_carbono'].tolist()[:qtd_imagens])
    if (args.debug):
        print(f'{prefix} Shape Y_train_carbono: {Y_train_carbono.shape}')
        
    #Y_train_nitrogenio = np.array(df_train['teor_nitrogenio'].tolist()[:qtd_imagens])
    #print(f'Shape Y_train_nitrogenio: {Y_train_nitrogenio.shape}')



    # *****************************************************
    # Modelo novo com GlobalAveragePooling2D
    # Parâmetros: sem o camada pooling, drop column teor_nitrogenio, layer.trainable = False, preproc = True
    # *****************************************************
    pretrained_model.trainable = args.trainable
    for layer in pretrained_model.layers:
        layer.trainable = args.trainable

    # Adicionando camadas personalizadas no topo do modelo
    x = pretrained_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    predictions = tf.keras.layers.Dense(1, activation='linear')(x)

    # Define o novo modelo combinando a ResNet50 com as camadas personalizadas
    model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=predictions)
        
    print(f'{prefix}')
    print(model.summary())
    print(f'{prefix}')

    # Otimizadores
    # https://keras.io/api/optimizers/
    # Usuais
    #  tf.keras.optimizers.Adam(learning_rate=0.0001)
    #  tf.keras.optimizers.RMSprop(learning_rate=0.0001)
    #  tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
    #  tf.keras.optimizers.Nadam(learning_rate=0.0001)
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
    
    model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mse'])
    history = model.fit(X_train, Y_train_carbono, validation_split=0.3, epochs=100, callbacks=[
                               tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)])

    model.save(args.name)
    
    print(f"{prefix} Info parameters: ")
    print(f"{prefix}{prefix} -d (--debug): {args.debug}")
    print(f"{prefix}{prefix} -n (--name): {args.name}")
    print(f"{prefix}{prefix} -p (--preprocess): {args.preprocess}")
    print(f"{prefix}{prefix} -t (--trainable): {args.trainable}")
