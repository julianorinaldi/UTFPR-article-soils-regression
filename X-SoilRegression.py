# -*- coding: utf-8 -*-

# Imports
import argparse

from modelSet import convertModelSet
from entityModelConfig import ModelConfig

prefix = ">>>>>>>>>>>>>>>>>"

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--debug", action="store_true", help="Para listar os prints de Debug")
parser.add_argument("-e", "--epochs", type=int, default=100, help="Quantidade de épocas para o treino - [Modelos TransferLearning/CNN]")
parser.add_argument("-G", "--grid_search", action="store_true", default=False, help="Treinar modelos com diversos hyperparametros (estabelecidos hardcoded) - [Todos Modelos]")
parser.add_argument("-i", "--amount_image_train", type=int, default=8930, help="Quantidade de imagens para treino")
parser.add_argument("-I", "--amount_image_test", type=int, default=3843, help="Quantidade de imagens para test")
parser.add_argument("-m", "--model", type=int, default=0, help="Modelo: [0]-ResNet50, [1]-ResNet101, [2]-ResNet152, [10]-ConvNeXtBase, [11]-ConvNeXtXLarge, [20]-EfficientNetB7, [21]-EfficientNetV2S, [22]-EfficientNetV2L, [30]-InceptionResNetV2, [40]-DenseNet169, [50]-VGG19, [100]-CNN, [500]-XGBRegressor, [510]-LinearRegression, [520]-SVMLinearRegression, [521]-SVMRBFRegressor")
parser.add_argument("-M", "--show_model", action="store_true", default=False, help="Mostra na tela os layers do modelo - [Modelos TransferLearning/CNN]")
parser.add_argument("-n", "--name", help="Nome do arquivo/diretório de saída do modelo .tf")
parser.add_argument("-p", "--preprocess", action="store_true", default=False, help="Preprocessar imagem 'model.preprocess_input(...)' - [Modelos TransferLearning]")
parser.add_argument("-P", "--patience", type=int, default=5, help="Quantidade de paciência no early stopping - [Modelos TransferLearning/CNN]")
parser.add_argument("-S", "--separed", action="store_true", default=False, help="Separar dados de treino e validação - [Modelos TransferLearning, CNN]")
parser.add_argument("-t", "--trainable", action="store_true", default=False, help="Define se terá as camadas do modelo de transfer-learning treináveis ou não - [Modelos TransferLearning]")
parser.add_argument("-T", "--Test", action="store_true", default=False, help="Define execução apenas para o teste - [Modelos TransferLearning]")


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
                          channelColors=qtd_canal_color, amountImagesTrain=args.amount_image_train,
                          amountImagesTest=args.amount_image_test,
                          argsNameModel=args.name,argsDebug=args.debug, argsTrainable=args.trainable,
                          argsSepared=args.separed, argsPreprocess=args.preprocess, argsOnlyTest=args.Test, 
                          argsEpochs=args.epochs, argsPatience=args.patience, argsGridSearch=args.grid_search,
                          argsShowModel=args.show_model,
                          printPrefix = prefix)

# Estratégia de importar a execução, para não carregar o TensorFlow antes de acetar parâmetros de entrada.
from XExecute import execute

execute(modelConfig)