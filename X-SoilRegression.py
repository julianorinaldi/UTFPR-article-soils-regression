# -*- coding: utf-8 -*-

# Imports
import argparse
import os

import numpy as np  # Trabalhar com array
import tensorflow as tf  # Trabalhar com aprendizado de máquinas

from tqdm import tqdm  # Facilita visualmente a iteração usado no "for"

from imageProcess import image_load, image_convert_array
from datasetProcess import dataset_process
from modelSet import ModelSet
from entityModelConfig import ModelConfig
from trainingCarbon import TrainingCarbon
from testingCarbon import TestCarbon
from modelTransferLearningProcess import modelTransferLearningProcess

prefix = ">>>>>>>>>>>>>>>>>"

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true", nargs='?', help="Para listar os prints de Debug")
parser.add_argument("-n", "--name", nargs='?', help="Nome do arquivo de saída do modelo .tf")
parser.add_argument("-p", "--preprocess", nargs='?', action="store_true", default=False, help="Preprocessar imagem 'model.preprocess_input(...)'")
parser.add_argument("-t", "--trainable", nargs='?', action="store_true", default=False, help="Define se terá as camadas do modelo de transfer-learning treináveis ou não")
parser.add_argument("-T", "--Test", nargs='?', action="store_true", default=False, help="Define execução apenas para o teste")
parser.add_argument("-e", "--epochs", nargs='?', action="count", default=100, type=int, help="Quantidade de épocas para o treino")
parser.add_argument("-P", "--patience", nargs='?', action="count", default=5, type=int, help="Quantidade de paciência no early stopping")

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
modelSet = ModelSet.EfficientNetV2S
imageDimensionX = 256
imageDimensionY = 256
qtd_canal_color = 3
pathCsv = ""
dir_base_img = ""
modelConfig = ModelConfig(modelSet=modelSet, pathCSV=pathCsv, dir_base_img=dir_base_img,
                          imageDimensionX=imageDimensionX, imageDimensionY=imageDimensionY,
                          channelColors=qtd_canal_color, 
                          argsNameModel=args.name,argsDebug=args.debug, argsTrainable=args.trainable,
                          argsPreprocess=args.preprocess, argsOnlyTest=args.Test, argsEpochs=args.epochs, 
                          argsPatience=args.patience,
                          printPrefix = prefix)


# Estratégia para trabalhar com Multi-GPU
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with strategy.scope():
    if (not modelConfig.argsOnlyTest):
        print()
        print(f'{modelConfig.printPrefix}')
        print(f'{modelConfig.printPrefix} Iniciando o Treino')
        print(f'{modelConfig.printPrefix}')
        modelConfig.setDirBaseImg('dataset/images/treinamento-solo-256x256')
        modelConfig.setPathCSV('dataset/csv/Dataset256x256-Treino.csv')
        training = TrainingCarbon(modelConfig)
        training.train()
    else:
        print()
        print(f'{modelConfig.printPrefix}')
        print(f'{modelConfig.printPrefix} Somente execução por Teste')
        print(f'{modelConfig.printPrefix}')
    
    print()
    print(f'{modelConfig.printPrefix}')
    print(f'{modelConfig.printPrefix} Iniciando o Teste')
    print(f'{modelConfig.printPrefix}')
    modelConfig.setDirBaseImg('dataset/images/teste-solo-256x256')
    modelConfig.setPathCSV('dataset/csv/Dataset256x256-Teste.csv')
    testCarbon = TestCarbon(modelConfig)
    testCarbon.test()
    
    print()
    print(f"{modelConfig.printPrefix} Info parameters: ")
    print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -d (--debug): {modelConfig.argsDebug}")
    print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -n (--name): {modelConfig.argsNameModel}")
    print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -p (--preprocess): {modelConfig.argsPreprocess}")
    print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -t (--trainable): {modelConfig.argsTrainable}")
    print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -T (--Test): {modelConfig.argsOnlyTest}")
    print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -e (--epochs): {modelConfig.argsEpochs}")
    print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -P (--patience): {modelConfig.argsPatience}")
    
    print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} Model: {modelConfig.modelSet.name}")