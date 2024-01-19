# -*- coding: utf-8 -*-

# Imports
import argparse
from log.LoggingPy import LoggingPy

from core.ModelSetEnum import convertModelSetEnum
from core.ModelConfig import ModelConfig
from core.ExecuteProcess import ExecuteProcess

prefix = ">>>>>>>>>>>>>>>>>"

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=100, help="Quantidade de épocas para o treino - [Modelos TransferLearning/CNN]")
parser.add_argument("-G", "--grid_search_trials", type=int, default=0, help="Treinar modelos com diversos hyperparametros (setar > 0 para rodar) - [Modelos TransferLearning/CNN hardcoded]")
parser.add_argument("-i", "--amount_image_train", type=int, default=8930, help="Quantidade de imagens para treino")
parser.add_argument("-I", "--amount_image_test", type=int, default=3843, help="Quantidade de imagens para test")
parser.add_argument("-L", "--log_level", type=int, default=1, help="Log Level: [0]-DEBUG, [1]-INFO - DEFAULT")
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
    print(f'{prefix} Há parâmetros faltantes. Utilize -h ou --help para ajuda!')
    exit(1)

logger = LoggingPy(args.name, prefix, args.log_level)
logger.logInfo(f"Parâmetros:")
logger.logInfo(args)

# Definindo Modelo de TransferLearning e Configurações
modelSetEnum = convertModelSetEnum(args.model)
imageDimensionX = 256
imageDimensionY = 256
qtd_canal_color = 3
pathCsv = ""
dir_base_img = ""

modelConfig = ModelConfig(modelSetEnum=modelSetEnum, loggingPy=logger, pathCSV=pathCsv, dir_base_img=dir_base_img,
                          imageDimensionX=imageDimensionX, imageDimensionY=imageDimensionY,
                          channelColors=qtd_canal_color, amountImagesTrain=args.amount_image_train,
                          amountImagesTest=args.amount_image_test, log_level=args.log_level,
                          argsNameModel=args.name, argsTrainable=args.trainable,
                          argsSepared=args.separed, argsPreprocess=args.preprocess, argsOnlyTest=args.Test, 
                          argsEpochs=args.epochs, argsPatience=args.patience, argsGridSearch=args.grid_search_trials,
                          argsShowModel=args.show_model)


# Estratégia de importar a execução, para não carregar o TensorFlow antes de acetar parâmetros de entrada.
#from core.ExecuteProcess import ExecuteProcess

execute = ExecuteProcess(modelConfig)
execute.run()