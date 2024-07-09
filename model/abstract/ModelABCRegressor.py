import pandas as pd
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from core.ResultLogger import ResultLogger
from processor.TestProcessor import TestProcessor
from dto.FitDTO import FitDTO
from dto.ConfigTrainModelDTO import ConfigTrainModelDTO
from core.DataProcessTrain import DataProcessTrain
from processor.ImageProcessor import ImageProcessor
from keras_tuner.tuners import Hyperband


class ModelABCRegressor(ABC):
    def __init__(self, config: ConfigTrainModelDTO):
        self.config = config
        self.models = []
        self.hyperparameters = []
        super().__init__()

    # Implemente para cada modelo de algoritmo de machine learn
    @abstractmethod
    def get_specialist_model(self, hp):
        pass

    # Implemente se não desejar converter em 2 dimensões
    # Padrão que vem: (qtdImage, 256,256,3)
    # Na implementação abaixo, fica: (qtdImage, 196608) usado para algoritmos padrões
    @abstractmethod
    def reshape_two_dimensions(self, x_data):
        return x_data.reshape(x_data.shape[0], -1)

    # Re-implemente se desejar fazer um fit diferente, por exempĺo para CNN
    @abstractmethod
    def model_fit(self, models, fit_dto: FitDTO):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_mae', patience=self.config.argsPatience,
            restore_best_weights=True)

        for model in models:
            if not self.config.argsSepared:
                # Juntando os dados de validação com treino no SUPER.
                x_img_data = np.concatenate((fit_dto.x_img_train, fit_dto.x_img_validate), axis=0)
                y_df_data = np.concatenate((fit_dto.y_df_train, fit_dto.y_df_validate), axis=0)
                model.fit(x_img_data, y_df_data, epochs=self.config.argsEpochs, callbacks=[early_stopping])
            else:
                model.fit(fit_dto.x_img_train, fit_dto.y_df_train,
                          validation_data=(fit_dto.x_img_validate, fit_dto.y_df_validate),
                          epochs=self.config.argsEpochs,
                          callbacks=[early_stopping])

    def __show_predict_samples(self, df_amostra, y_df_data, prediction):
        result_logger = ResultLogger(self.config.logger)
        result_logger.logger_predict_test(df_amostra, y_df_data, prediction)

    def __execute_grid_search(self, fit_dto: FitDTO):
        # Executa com GridSearch
        self.config.logger.log_info(f"Executando [COM] o GridSearch ...")
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mae', patience=self.config.argsPatience, restore_best_weights=True)

        tuner = Hyperband(
             hypermodel=self.get_specialist_model,
             objective='val_mae',
             max_epochs=self.config.argsEpochs,
             factor=3,
             max_trials=self.config.argsGridSearch,  # Quantas tentativas de hiperparâmetros serão executadas
             hyperband_iterations=self.config.argsGridSearch,
             directory='_hyperbandResults',  # diretório para armazenar os resultados
             project_name=self.config.argsNameModel
        )

        if not self.config.argsSepared:
            # Padrão sem separação entre validação e treino
            x_img_train = np.concatenate((fit_dto.x_img_train, fit_dto.x_img_validate), axis=0)
            y_train = np.concatenate((fit_dto.y_df_train, fit_dto.y_df_validate), axis=0)
            tuner.search(x_img_train, y_train, epochs=self.config.argsEpochs,
                         validation_split = 0.3, callbacks = [early_stopping])
        else:
            # Execute a busca de hiperparâmetros
            tuner.search(fit_dto.x_img_train, fit_dto.y_df_train, epochs=self.config.argsEpochs,
                         validation_data = (fit_dto.x_img_validate, fit_dto.y_df_validate),
                         callbacks = [early_stopping])

            self.hyperparameters = tuner.get_best_hyperparameters(num_trials=self.config.argsGridSearch)
            # Imprima os melhores hiperparâmetros encontrados
            self.config.logger.log_info(f"Melhores Hyperparameters:")
            self.config.logger.log_info(f"{self.hyperparameters[0].values}")

            # Obtenha a melhor tentativa
            best_trial = tuner.oracle.get_best_trials(num_trials=self.config.argsGridSearch)
            _models = []
            for trial in best_trial:
                _models.append(tuner.load_model(trial))
            self.models = _models
            del tuner

    def train(self, df_train: pd.DataFrame):
        self.config.logger.log_info("Iniciando o Treinamento ...")
        self.config.logger.log_info("Separando Treino e Validação ...")
        dp = DataProcessTrain(self.config)
        # Separar dados de Treinamento e de Validação
        y_df_train, y_df_validate = dp.get_train_validate_process(df_train)

        # 80% para Treino
        amount_img_train: int = round(self.config.amountImagesTrain * 0.80)
        if amount_img_train > len(y_df_train):
            amount_img_train = len(y_df_train)
        # 20% para Validação
        amount_img_validate: int = round(self.config.amountImagesTrain * 0.20)
        if amount_img_validate > len(y_df_validate):
            amount_img_validate = len(y_df_validate)
        self.config.logger.log_info("Quantidade de imagens Treino/Validação: %d/%d" % (amount_img_train, amount_img_validate))

        # Carregar imagens e separar em treino e validação
        self.config.logger.log_info(f"Carregando imagens de Treino ...")
        img_p = ImageProcessor(self.config.get_image_config())
        x_img_train_array = img_p.image_load(df=y_df_train, path_img=DataProcessTrain.PATH_IMG_TRAIN, amount_img=amount_img_train)
        x_img_validate_array = img_p.image_load(df=y_df_validate, path_img=DataProcessTrain.PATH_IMG_TRAIN, amount_img=amount_img_validate)

        # Remover a coluna 'arquivo' após carregamento das imagens
        y_df_train = y_df_train.drop(columns=["arquivo"])
        y_df_validate = y_df_validate.drop(columns=["arquivo"])

        # Limitar quantidade de Dados de acordo com Quantidade de Imagens
        self.config.logger.log_debug(f"Limitando Dados ...")
        y_df_train = y_df_train.head(amount_img_train)
        y_df_validate = y_df_validate.head(amount_img_validate)
        self.config.logger.log_debug(f"Imagens no Treino no X: {len(x_img_train_array)} - "
                                     f"Imagens na Validade no X: {len(x_img_validate_array)}\n"
                                     f"Shape no Treino: {y_df_train.shape} - "
                                     f"Shape na Validade: {y_df_validate.shape}\n")

        self.config.logger.log_info(f"Resumo Dados Treino:\n{y_df_train.describe()}\n")
        self.config.logger.log_info(f"Resumo Dados Validate:\n{y_df_validate.describe()}\n")
        # Aceita apenas 2 dimensões.
        #y_df_train = self.reshape_two_dimensions(y_df_train)
        #y_df_validate = self.reshape_two_dimensions(y_df_validate)

        # Treinar o modelo
        self.config.logger.log_info(f"Iniciando o treino ...")

        fit_dto: FitDTO = FitDTO(x_img_train_array, y_df_train, x_img_validate_array, y_df_validate)
        if not self.config.argsGridSearch > 0:
            # Executa sem GridSearch
            model = self.get_specialist_model(hp=None)
            self.models = [model]
            self.config.logger.log_info(f"Executando [SEM] o GridSearch ...")
            self.model_fit(self.models, fit_dto)
        else:
            self.__execute_grid_search(fit_dto)

    def test(self, df_test: pd.DataFrame):
        test_processor: TestProcessor = TestProcessor(self.config.get_config_test())
        test_processor.test(df_test, self.models)
