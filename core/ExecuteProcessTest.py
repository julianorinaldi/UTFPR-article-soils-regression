import os
import tensorflow as tf

from core.DataProcessTest import DataProcessTest
from core.ExecuteProcessBase import ExecuteProcessBase
from processor.TestProcessor import TestProcessor
from dto.ConfigTestDTO import ConfigTestDTO


# Defina o nível de log para suprimir mensagens de nível INFO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
class ExecuteProcessTest(ExecuteProcessBase):
    def __init__(self, config: ConfigTestDTO):
        super().__init__(config.logger)
        self.config = config

    def run(self):
        # Cria a estrategia de paralelismo
        strategy = self._get_gpu_strategy()
        # Cria o modelo
        _model = self.__get_model_instance()

        dp = DataProcessTest(self.config)
        df_test = dp.load_test_data()

        with strategy.scope():
            test_processor: TestProcessor = TestProcessor(self.config)
            test_processor.test(df_test, [_model])

    def __get_model_instance(self) -> tf.keras.models.Model:
        resnet_model = tf.keras.models.load_model(filepath = self.config.argsNameModel)
        if self.config.argsShowModel:
            self.logger.log_info(f"{resnet_model.summary()}")
        else:
            self.logger.log_debug(f"{resnet_model.summary()}")
        return resnet_model