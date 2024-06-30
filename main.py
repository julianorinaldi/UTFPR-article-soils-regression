# -*- coding: utf-8 -*-

# Imports
import argparse

from core.ArgsProcessor import get_config_from_args
from shared.infrastructure.log.LoggingPy import LoggingPy
from dto.ConfigModelDTO import ConfigModelDTO


def main():
    args = get_args()

    logger = LoggingPy(name_model=args.name, log_level=args.log_level)
    try:
        config = get_config_from_args(args, logger)
    except Exception as e:
        logger.log_error(e)
        exit(1)

    logger.log_info(f"Parâmetros:")
    logger.log_info(args)

    try:
        execute_process(config)
    except Exception as e:
        logger.log_error(e)
        exit(1)


def get_args():
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

    return parser.parse_args()

def execute_process(config: ConfigModelDTO):
    # Estratégia de importar a execução, para não carregar o TensorFlow antes de acetar parâmetros de entrada.
    from core.ExecuteProcess import ExecuteProcess
    execute = ExecuteProcess(config)
    execute.run()
    info_args(config)


def info_args(config):
    config.logger.log_info("#######################")
    config.logger.log_info(f"Info parameters: ")
    config.logger.log_info(f" -e (--epochs): {config.argsEpochs}")
    config.logger.log_info(f" -G (--grid_search_trials): {config.argsGridSearch}")
    config.logger.log_info(f" -i (--amount_image_train): {config.amountImagesTrain}")
    config.logger.log_info(f" -I (--amount_image_test): {config.amountImagesTest}")
    config.logger.log_info(f" -L (--log_level): {config.log_level}")
    config.logger.log_info(
        f" -m (--model): {config.modelSetEnum.value} - {config.modelSetEnum.name}")
    config.logger.log_info(f" -M (--show_model): {config.argsShowModel}")
    config.logger.log_info(f" -n (--name): {config.argsNameModel}")
    config.logger.log_info(f" -N (--normalize): {config.argsNormalize}")
    config.logger.log_info(f" -p (--preprocess): {config.argsPreprocess}")
    config.logger.log_info(f" -P (--patience): {config.argsPatience}")
    config.logger.log_info(f" -S (--separed): {config.argsSepared}")
    config.logger.log_info(f" -t (--trainable): {config.argsTrainable}")
    config.logger.log_info(f" -T (--Test): {config.argsOnlyTest}")
    config.logger.log_info("#######################")

if __name__ == "__main__":
    main()