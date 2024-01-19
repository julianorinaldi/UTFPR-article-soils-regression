import os
# Defina o nível de log para suprimir mensagens de nível INFO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf  # Trabalhar com aprendizado de máquinas

# Models
from model.ModelCNNRegressor import ModelRegressorCNN
from model.ModelXGBRegressor import ModelXGBRegressor
from model.ModelLinearRegressor import ModelLinearRegressor
from model.ModelSVMLinearRegressor import ModelSVMLinearRegressor
from model.ModelSVMRBFRegressor import ModelSVMRBFRegressor
from model.ModelTransferLearningRegressor import ModelRegressorTransferLearning

from log.LoggingPy import LoggingPy
from core.ModelConfig import ModelConfig
from core.ModelSetEnum import ModelSetEnum

class ExecuteProcess:
    def __init__(self, config : ModelConfig):
        self.config = config

    def run(self):
        physical_devices = tf.config.list_physical_devices('GPU')

        # Infos da GPU e Framework
        if (self.config.argsDebug):
            print(f'{self.config.printPrefix} Amount of GPU Available: {physical_devices}')

        # Estratégia para trabalhar com Multi-GPU
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=2))
        
        print(f'{self.config.printPrefix} Modelo: {self.config.modelSetEnum.name}')
        with strategy.scope():
            if (self.config.modelSetEnum == ModelSetEnum.ResNet50 or
                self.config.modelSetEnum == ModelSetEnum.ResNet101 or
                self.config.modelSetEnum == ModelSetEnum.ResNet152 or
                self.config.modelSetEnum == ModelSetEnum.EfficientNetB7 or
                self.config.modelSetEnum == ModelSetEnum.EfficientNetV2S or
                self.config.modelSetEnum == ModelSetEnum.EfficientNetV2L or
                self.config.modelSetEnum == ModelSetEnum.ConvNeXtBase or
                self.config.modelSetEnum == ModelSetEnum.ConvNeXtXLarge or
                self.config.modelSetEnum == ModelSetEnum.DenseNet169 or
                self.config.modelSetEnum == ModelSetEnum.VGG19 or
                self.config.modelSetEnum == ModelSetEnum.InceptionResNetV2):
                _model = ModelRegressorTransferLearning(self.config)
        
            elif (self.config.modelSetEnum == ModelSetEnum.CNN):
                _model = ModelRegressorCNN(self.config)
        
            elif (self.config.modelSetEnum == ModelSetEnum.XGBRegressor):
                _model = ModelXGBRegressor(self.config)

            elif (self.config.modelSetEnum == ModelSetEnum.LinearRegression):
                _model = ModelLinearRegressor(self.config)

            elif (self.config.modelSetEnum == ModelSetEnum.SVMLinearRegression):
                _model = ModelSVMLinearRegressor(self.config)

            elif (self.config.modelSetEnum == ModelSetEnum.SVMRBFRegressor):
                _model = ModelSVMRBFRegressor(self.config)

            else:
                raise Exception('Modelo desconhecido')
            
            _model.train()
            _model.test()
            
            print()
            print(f"{self.config.printPrefix} Info parameters: ")
            print(f"{self.config.printPrefix}{self.config.printPrefix} -d (--debug): {self.config.argsDebug}")
            print(f"{self.config.printPrefix}{self.config.printPrefix} -n (--name): {self.config.argsNameModel}")
            print(f"{self.config.printPrefix}{self.config.printPrefix} -p (--preprocess): {self.config.argsPreprocess}")
            print(f"{self.config.printPrefix}{self.config.printPrefix} -t (--trainable): {self.config.argsTrainable}")
            print(f"{self.config.printPrefix}{self.config.printPrefix} -T (--Test): {self.config.argsOnlyTest}")
            print(f"{self.config.printPrefix}{self.config.printPrefix} -e (--epochs): {self.config.argsEpochs}")
            print(f"{self.config.printPrefix}{self.config.printPrefix} -P (--patience): {self.config.argsPatience}")
            print(f"{self.config.printPrefix}{self.config.printPrefix} -i (--amount_image_train): {self.config.amountImagesTrain}")
            print(f"{self.config.printPrefix}{self.config.printPrefix} -I (--amount_image_test): {self.config.amountImagesTest}")
            print(f"{self.config.printPrefix}{self.config.printPrefix} -m (--model): {self.config.modelSetEnum.value} - {self.config.modelSetEnum.name}")
            print()