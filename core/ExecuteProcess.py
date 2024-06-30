import os
import tensorflow as tf  # Trabalhar com aprendizado de máquinas

# Models
from model.ModelCNNRegressor import ModelRegressorCNN
from model.ModelXGBRegressor import ModelXGBRegressor
from model.ModelLinearRegressor import ModelLinearRegressor
from model.ModelSVMLinearRegressor import ModelSVMLinearRegressor
from model.ModelSVMRBFRegressor import ModelSVMRBFRegressor
from model.ModelTransferLearningRegressor import ModelRegressorTransferLearning
from model.ModelPLSRegression import ModelPLSRegression
from dto.ConfigModelDTO import ConfigModelDTO
from dto.ModelSetEnum import ModelSetEnum

# Defina o nível de log para suprimir mensagens de nível INFO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
class ExecuteProcess:
    def __init__(self, config: ConfigModelDTO):
        self.config = config

    # Cria a estrategia de paralelismo
    def __get_gpu_strategy(self):
        physical_devices = tf.config.list_physical_devices('GPU')
        # Infos da GPU e Framework
        self.config.logger.log_debug(f"Quantidade de GPU disponíveis: {physical_devices}")

        # Estratégia para trabalhar com Multi-GPU
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce(num_packs=2))

        return strategy

    def run(self):
        # Cria a estrategia de paralelismo
        strategy = self.__get_gpu_strategy()
        # Cria o modelo
        _model = self.__get_model_instance()

        with strategy.scope():
            # Chama o treinamento
            _model.train()
            # Chama os testes
            _model.test()

    def __get_model_instance(self):
        self.config.logger.log_info(f"Modelo: {self.config.modelSetEnum.name}")
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
        elif self.config.modelSetEnum == ModelSetEnum.CNN:
            _model = ModelRegressorCNN(self.config)
        elif self.config.modelSetEnum == ModelSetEnum.XGBRegressor:
            _model = ModelXGBRegressor(self.config)
        elif self.config.modelSetEnum == ModelSetEnum.LinearRegression:
            _model = ModelLinearRegressor(self.config)
        elif self.config.modelSetEnum == ModelSetEnum.SVMLinearRegression:
            _model = ModelSVMLinearRegressor(self.config)
        elif self.config.modelSetEnum == ModelSetEnum.SVMRBFRegressor:
            _model = ModelSVMRBFRegressor(self.config)
        elif self.config.modelSetEnum == ModelSetEnum.PLSRegression:
            _model = ModelPLSRegression(self.config)
        else:
            raise Exception("Modelo desconhecido")
        return _model