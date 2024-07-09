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

    def __find_file_in_subdirectories(self, filename: str, start_directory="."):
        for root, dirs, files in os.walk(start_directory):
            if filename in files:
                filename_path = os.path.abspath(os.path.join(root, filename))
                self.logger.log_debug(f"Modelo: {filename_path}")
                return filename_path
        return None

    def __get_model_instance(self) -> tf.keras.models.Model:
        file_path = self.__find_file_in_subdirectories(self.config.argsNameModel)
        if file_path:
            resnet_model = tf.keras.models.load_model(filepath = file_path)
            if self.config.argsShowModel:
                self.logger.log_info(f"{resnet_model.summary()}")
            else:
                self.logger.log_debug(f"{resnet_model.summary()}")
            return resnet_model
        else:
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
