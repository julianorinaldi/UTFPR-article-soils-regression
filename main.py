# -*- coding: utf-8 -*-

# Imports
import argparse

from core.NormalizeEnum import convert_normalize_set_enum
from log.LoggingPy import LoggingPy

from core.ModelSetEnum import convert_model_set_enum
from core.ModelConfig import ModelConfig

prefix = ">>>>>"

# Argumentos
parser = argparse.ArgumentParser()
parser.add_argument("-e", "--epochs", type=int, default=100,
                    help="Quantidade de épocas para o treino - [Modelos TransferLearning/CNN]")
parser.add_argument("-G", "--grid_search_trials", type=int, default=0,
                    help="Treinar modelos com diversos hyperparametros (setar > 0 para rodar) - [Modelos "
                         "TransferLearning/CNN hardcoded]")
parser.add_argument("-i", "--amount_image_train", type=int, default=8930, help="Quantidade de imagens para treino")
parser.add_argument("-I", "--amount_image_test", type=int, default=3843, help="Quantidade de imagens para test")
parser.add_argument("-L", "--log_level", type=int, default=1, help="Log Level: [0]-DEBUG, [1]-INFO - DEFAULT")
parser.add_argument("-m", "--model", type=int, default=0, help="Modelo: [0]-ResNet50, [1]-ResNet101, [2]-ResNet152, "
                                                               "[10]-ConvNeXtBase, [11]-ConvNeXtXLarge, "
                                                               "[20]-EfficientNetB7, [21]-EfficientNetV2S, "
                                                               "[22]-EfficientNetV2L, [30]-InceptionResNetV2, "
                                                               "[40]-DenseNet169, [50]-VGG19, [100]-CNN, "
                                                               "[200]-PLSRegression, [500]-XGBRegressor, "
                                                               "[510]-LinearRegression, [520]-SVMLinearRegression, "
                                                               "[521]-SVMRBFRegressor")
parser.add_argument("-M", "--show_model", action="store_true", default=False, help="Mostra na tela os layers do "
                                                                                   "modelo - [Modelos "
                                                                                   "TransferLearning/CNN]")
parser.add_argument("-n", "--name", help="Nome do arquivo/diretório de saída do modelo .tf")
parser.add_argument("-N", "--normalize", type=int, default=0, help="Normalizar dados das imagens: [0]-Default não "
                                                                   "normalizado, [1]-MinMaxScaler, [2]-RobustScaler, "
                                                                   "[3]-StandardScaler, [4]-Z-score")
parser.add_argument("-p", "--preprocess", action="store_true", default=False, help="Preprocessar imagem "
                                                                                   "'model.preprocess_input(...)' - ["
                                                                                   "Modelos TransferLearning]")
parser.add_argument("-P", "--patience", type=int, default=5, help="Quantidade de paciência no early stopping - ["
                                                                  "Modelos TransferLearning/CNN]")
parser.add_argument("-S", "--separed", action="store_true", default=False, help="Separar dados de treino e validação "
                                                                                "- [Modelos TransferLearning, CNN]")
parser.add_argument("-t", "--trainable", action="store_true", default=False, help="Define se terá as camadas do "
                                                                                  "modelo de transfer-learning "
                                                                                  "treináveis ou não - [Modelos "
                                                                                  "TransferLearning]")
parser.add_argument("-T", "--Test", action="store_true", default=False, help="Define execução apenas para o teste - ["
                                                                             "Modelos TransferLearning]")

args = parser.parse_args()

if not args.name:
    print(f'{prefix} Há parâmetros faltantes. Utilize -h ou --help para ajuda!')
    exit(1)

logger = LoggingPy(args.name, prefix, args.log_level)
logger.log_info(f"Parâmetros:")
logger.log_info(args)

# Definindo Modelo de TransferLearning e Configurações
modelSetEnum = convert_model_set_enum(args.model)
imageDimensionX = 256
imageDimensionY = 256
qtd_canal_color = 3
pathCsv = ""
dir_base_img = ""
normalizeEnum = convert_normalize_set_enum(args.normalize)

config = ModelConfig(model_set_enum=modelSetEnum, path_csv=pathCsv, dir_base_img=dir_base_img,
                     image_dimension_x=imageDimensionX, image_dimension_y=imageDimensionY,
                     channel_colors=qtd_canal_color, amount_images_train=args.amount_image_train,
                     amount_images_test=args.amount_image_test, log_level=args.log_level,
                     args_name_model=args.name,args_normalize=normalizeEnum, args_trainable=args.trainable,
                     args_separed=args.separed, args_preprocess=args.preprocess, args_only_test=args.Test,
                     args_epochs=args.epochs, args_patience=args.patience, args_grid_search=args.grid_search_trials,
                     args_show_model=args.show_model)
config.set_logger(logger)

# Estratégia de importar a execução, para não carregar o TensorFlow antes de acetar parâmetros de entrada.
from core.ExecuteProcess import ExecuteProcess

execute = ExecuteProcess(config)
execute.run()
