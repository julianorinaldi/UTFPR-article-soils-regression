import os
import tensorflow as tf

from core.DataProcessTrain import DataProcessTrain
from core.ExecuteProcessBase import ExecuteProcessBase
from model.ModelCNNRegressor import ModelRegressorCNN
from model.ModelXGBRegressor import ModelXGBRegressor
from model.ModelLinearRegressor import ModelLinearRegressor
from model.ModelSVMLinearRegressor import ModelSVMLinearRegressor
from model.ModelSVMRBFRegressor import ModelSVMRBFRegressor
from model.ModelTransferLearningRegressor import ModelRegressorTransferLearning
from model.ModelPLSRegression import ModelPLSRegression
from dto.ConfigTrainModelDTO import ConfigTrainModelDTO
from dto.ModelSetEnum import ModelSetEnum
from model.abstract.ModelABCRegressor import ModelABCRegressor

# Defina o nível de log para suprimir mensagens de nível INFO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
class ExecuteProcessTrain(ExecuteProcessBase):
    def __init__(self, config: ConfigTrainModelDTO):
        super().__init__(config.logger)
        self.config = config

    def run(self):
        strategy = self._get_gpu_strategy()
        _model = self.__get_model_instance()

        dp = DataProcessTrain(self.config)
        df_train, df_test = dp.load_train_test_data()

        with strategy.scope():
            # Chama o treinamento
            _model.train(df_train)
            # Chama os testes
            _model.test(df_test)

    def __get_model_instance(self) -> ModelABCRegressor:
        self.logger.log_info(f"Modelo: {self.config.modelSetEnum.name}")
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