# -*- coding: utf-8 -*-

# Imports
import argparse

from modelSet import convertModelSet
from entityModelConfig import ModelConfig

prefix = ">>>>>>>>>>>>>>>>>"

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true", help="Para listar os prints de Debug")
parser.add_argument("-n", "--name", help="Nome do arquivo de saída do modelo .tf")
parser.add_argument("-p", "--preprocess", action="store_true", default=False, help="Preprocessar imagem 'model.preprocess_input(...)'")
parser.add_argument("-t", "--trainable", action="store_true", default=False, help="Define se terá as camadas do modelo de transfer-learning treináveis ou não")
parser.add_argument("-T", "--Test", action="store_true", default=False, help="Define execução apenas para o teste")
parser.add_argument("-e", "--epochs", type=int, default=100, help="Quantidade de épocas para o treino")
parser.add_argument("-P", "--patience", type=int, default=5, help="Quantidade de paciência no early stopping")
parser.add_argument("-m", "--model", type=int, default=0, help="Selecione o modelo: 0 - ResNet50, 1 - ResNet101, 2 - ResNet152, 3 - ConvNeXtBase, 4- EfficientNetB7, 5 - EfficientNetV2S, 6 - InceptionResNetV2, 7 - DenseNet169, 8 - VGG19")

args = parser.parse_args()

if not (args.name):
    print(f'{prefix}')
    print(f'{prefix} Há parâmetros faltantes. Utilize -h ou --help para ajuda!')
    print(f'{prefix}')
    exit(1)

# Definindo Modelo de TransferLearning e Configurações
modelSet = convertModelSet(args.model)
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

from execProcess import executeProcess

executeProcess(modelConfig)