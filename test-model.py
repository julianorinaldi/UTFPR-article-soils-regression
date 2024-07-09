# -*- coding: utf-8 -*-

# Imports
import argparse

from processor.ArgsProcessor import get_config_test_from_args
from dto.ConfigTestDTO import ConfigTestDTO
from shared.infrastructure.log.LoggingPy import LoggingPy


def test():
    args = get_args()

    logger = LoggingPy(name_model=args.name, log_level=args.log_level)
    try:
        config = get_config_test_from_args(args, logger)
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
    parser.add_argument("-p", "--preprocess", action="store_true", default=False, help="Preprocessar imagem "
                                                                                       "'model.preprocess_input(...)' - ["
                                                                                       "Modelos TransferLearning]")
    return parser.parse_args()

def execute_process(config: ConfigTestDTO):
    # Estratégia de importar a execução, para não carregar o TensorFlow antes de acetar parâmetros de entrada.
    from core.ExecuteProcessTest import ExecuteProcessTest
    execute = ExecuteProcessTest(config)
    execute.run()
    info_args(config)


def info_args(config):
    config.logger.log_resume("#######################")
    config.logger.log_resume(f"Info parameters: ")
    config.logger.log_resume(f" -I (--amount_image_test): {config.amountImagesTest}")
    config.logger.log_resume(f" -L (--log_level): {config.log_level}")
    config.logger.log_resume(f" -M (--show_model): {config.argsShowModel}")
    config.logger.log_resume(f" -n (--name): {config.argsNameModel}")
    config.logger.log_resume(f" -p (--preprocess): {config.argsPreprocess}")
    config.logger.log_resume("#######################")

if __name__ == "__main__":
    test()