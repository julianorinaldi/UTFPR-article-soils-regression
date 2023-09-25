# -*- coding: utf-8 -*-

# Imports
import argparse
import os

import numpy as np  # Trabalhar com array
import tensorflow as tf  # Trabalhar com aprendizado de máquinas
from tqdm import tqdm  # Facilita visualmente a iteração usado no "for"

from imageProcess import image_processing
from datasetProcess import dataset_process
from modelSet import ModelSet
from entityModelConfig import ModelConfig
from modelTransferLearningProcess import modelTransferLearningProcess

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


# Definindo Modelo de TransferLearning e Configurações
modelSet = ModelSet.ResNet152
imageDimensionX = 256
imageDimensionY = 256
qtd_canal_color = 3
dir_base_img = 'dataset/images/treinamento-solo-256x256'
pathCsv = 'dataset/csv/Dataset256x256-Treino.csv'
modelConfig = ModelConfig(modelSet, pathCsv, dir_base_img,imageDimensionX, imageDimensionY, qtd_canal_color,
                          args.name, args.debug, args.trainable, args.preprocess)


# Estratégia para trabalhar com Multi-GPU
strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with strategy.scope():

    df_train, train_imagefiles = dataset_process(modelConfig)

    # Quantidade de imagens usadas para a rede.
    qtd_imagens = len(df_train)
    if (args.debug):
        print(f'{prefix} Preprocess: {args.preprocess}')
    
    # Array com as imagens a serem carregadas de treino
    image_list_train = []    
    for imageName in tqdm(train_imagefiles.tolist()[:qtd_imagens]):
        image_list_train.append(image_processing(modelConfig, imageName))

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

    # Faz a chamada da criação do modelo de Transferência
    pretrained_model = modelTransferLearningProcess(modelConfig)
    
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
