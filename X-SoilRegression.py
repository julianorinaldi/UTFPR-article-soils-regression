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
parser.add_argument("-d", "--debug", action="store_true", help="Para listar os prints de Debug")
parser.add_argument("-n", "--name", help="Nome do arquivo de saída do modelo .h5")
parser.add_argument("-p", "--preprocess", action="store_true", help="Preprocessar imagem 'resnet50.preprocess_input(...)'")
parser.add_argument("-t", "--trainable", action="store_true", help="Define se terá as camadas do modelo de transfer-learning treináveis ou não")

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
                          args.name, args.debug, args.trainable, args.preprocess, printPrefix = prefix)


# Estratégia para trabalhar com Multi-GPU
strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

with strategy.scope():
    modelConfig.dirBaseImg = 'dataset/images/treinamento-solo-256x256'
    modelConfig.pathCsv = 'dataset/csv/Dataset256x256-Treino.csv'
    training = TrainingCarbon(modelConfig)
    training.train()
    
    modelConfig.dirBaseImg = 'dataset/images/teste-solo-256x256'
    modelConfig.pathCsv = 'dataset/csv/Dataset256x256-Teste.csv'
    testCarbon = TestCarbon(modelConfig)
    testCarbon.test()
    
    print()
    print(f"{modelConfig.printPrefix} Info parameters: ")
    print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -d (--debug): {modelConfig.argsDebug}")
    print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -n (--name): {modelConfig.argsNameModel}")
    print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -p (--preprocess): {modelConfig.argsPreprocess}")