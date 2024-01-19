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

from core.ModelConfig import ModelConfig
from core.ModelSetEnum import ModelSetEnum

class ExecuteProcess:
    def __init__(self, config : ModelConfig):
        self.config = config

    def run(self):
        physical_devices = tf.config.list_physical_devices('GPU')

        # Infos da GPU e Framework
        self.config.logger.logDebug(f"Quantidade de GPU disponíveis: {physical_devices}")

        # Estratégia para trabalhar com Multi-GPU
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=2))
        
        self.config.logger.logInfo(f"Modelo: {self.config.modelSetEnum.name}")
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
                error = "Modelo desconhecido"
                self.config.logger.logInfo(f"Excetion: {error}")
                raise Exception(error)
            
            # Chama o treinamento
            _model.train()
            
            # Chama os testes
            _model.test()
            
            self.config.logger.logInfo("#######################")
            self.config.logger.logInfo(f"Info parameters: ")
            self.config.logger.logInfo(f" -e (--epochs): {self.config.argsEpochs}")
            self.config.logger.logInfo(f" -G (--grid_search_trials): {self.config.argsGridSearch}")
            self.config.logger.logInfo(f" -i (--amount_image_train): {self.config.amountImagesTrain}")
            self.config.logger.logInfo(f" -I (--amount_image_test): {self.config.amountImagesTest}")
            self.config.logger.logInfo(f" -L (--log_level): {self.config.log_level}")
            self.config.logger.logInfo(f" -m (--model): {self.config.modelSetEnum.value} - {self.config.modelSetEnum.name}")
            self.config.logger.logInfo(f" -M (--show_model): {self.config.argsShowModel}")
            self.config.logger.logInfo(f" -n (--name): {self.config.argsNameModel}")
            self.config.logger.logInfo(f" -p (--preprocess): {self.config.argsPreprocess}")
            self.config.logger.logInfo(f" -P (--patience): {self.config.argsPatience}")
            self.config.logger.logInfo(f" -S (--separed): {self.config.argsSepared}")
            self.config.logger.logInfo(f" -t (--trainable): {self.config.argsTrainable}")
            self.config.logger.logInfo(f" -T (--Test): {self.config.argsOnlyTest}")
            self.config.logger.logInfo("#######################")
            