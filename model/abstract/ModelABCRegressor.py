import pandas as pd
import numpy as np
import tensorflow as tf
from pandas import DataFrame
from tqdm import tqdm
from abc import ABC, abstractmethod

from dto.FitDTO import FitDTO
from dto.ConfigModelDTO import ConfigModelDTO
from core.DatasetProcess import DatasetProcess
from core.ImageProcess import ImageProcess
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from keras_tuner.tuners import Hyperband

from dto.TrainTestImgDTO import TrainTestImgDTO


class ModelABCRegressor(ABC):
    def __init__(self, config: ConfigModelDTO):
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
        self._min_max_predict_test(df_amostra, y_df_data, prediction)

    def __load_images(self, df: pd.DataFrame, amount_img: int) -> list:
        # Quantidade de imagens usadas para a rede.
        qtd_imagens = len(df)
        if (qtd_imagens > amount_img) and (amount_img > 0):
            qtd_imagens = amount_img

        img_p = ImageProcess(self.config)
        img_loaded = img_p.image_load(df, qtd_imagens)

        return  img_loaded

    def __execute_grid_search(self, fit_dto: FitDTO):
        # Executa com GridSearch
        self.config.logger.log_info(f"\nExecutando com o GridSearch\n")
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

    def train(self):
        self.config.set_dir_base_img('dataset/images/treinamento-solo-256x256')
        self.config.set_path_csv('dataset/csv/Dataset256x256-Treino.csv')
        self.config.logger.log_info("Iniciando o Treinamento...\n")
        # Tratamento inicial dos dados
        dp = DatasetProcess(self.config)
        df_all: pd.DataFrame = dp.load_and_clean_data

        self.config.logger.log_info("Separando Treino e Validação...\n")
        # Separar dados de Treinamento e de Validação
        y_df_train, y_df_validate = dp.get_train_validate_process(df_all)

        # 80% para Treino
        amount_img_train: int = round(self.config.amountImagesTrain * 0.80)
        # 20% para Validação
        amount_img_validate: int = round(self.config.amountImagesTrain * 0.20)
        self.config.logger.log_info("Quantidade de Imagens para Treino e Validação: %d, %d\n" % (amount_img_train, amount_img_validate))

        # Carregar imagens e separar em treino e validação
        self.config.logger.log_info(f"Carregando imagens ...\n")
        x_img_train_array = self.__load_images(df=y_df_train, amount_img=amount_img_train)
        x_img_validate_array = self.__load_images(df=y_df_validate, amount_img=amount_img_validate)

        # Remover a coluna 'arquivo' após carregamento das imagens
        y_df_train = y_df_train.drop(columns=["arquivo"])
        y_df_validate = y_df_validate.drop(columns=["arquivo"])

        # Limitar quantidade de Dados de acordo com Quantidade de Imagens
        self.config.logger.log_debug(f"Limitando Dados:\n")
        y_df_train = y_df_train.head(amount_img_train)
        y_df_validate = y_df_validate.head(amount_img_validate)
        self.config.logger.log_debug(f"\nTrain X: {len(x_img_train_array)}"
                                     f"\nValidate X: {len(x_img_validate_array)}"
                                     f"\nTrain Y: {y_df_train.shape}"
                                     f"\nValidate Y: {y_df_validate.shape}")

        self.config.logger.log_info(f"DataFrame Info Treino:\n{y_df_train.describe()}\n")
        self.config.logger.log_info(f"DataFrame Info Validate:\n{y_df_validate.describe()}\n")
        # Aceita apenas 2 dimensões.
        #y_df_train = self.reshape_two_dimensions(y_df_train)
        #y_df_validate = self.reshape_two_dimensions(y_df_validate)

        # Treinar o modelo
        self.config.logger.log_info(f"Iniciando o treino")

        fit_dto: FitDTO = FitDTO(x_img_train_array, y_df_train, x_img_validate_array, y_df_validate)
        if not self.config.argsGridSearch > 0:
            # Executa sem GridSearch
            model = self.get_specialist_model(hp=None)
            self.models = [model]
            self.config.logger.log_info(f"Executando sem o GridSearch\n")
            self.model_fit(self.models, fit_dto)
        else:
            self.__execute_grid_search(fit_dto)

    def test(self):
        # Agora entra o Test
        self.config.set_dir_base_img('dataset/images/teste-solo-256x256')
        self.config.set_path_csv('dataset/csv/Dataset256x256-Teste.csv')

        dp = DatasetProcess(self.config)
        y_df_all_test: pd.DataFrame = dp.load_and_clean_data
        df_amostra = y_df_all_test["amostra"]
        y_df_all_test = dp.get_test_process(df_all=y_df_all_test)

        x_img_test_array = self.__load_images(df=y_df_all_test, amount_img=self.config.amountImagesTest)
        x_img_test_array = tf.stack(x_img_test_array, axis=0)

        # Remover a coluna 'arquivo' para carregar as imagens
        y_df_all_test = y_df_all_test.drop(columns=["arquivo"])
        y_df_all_test = y_df_all_test.head(self.config.amountImagesTest)
        df_amostra = df_amostra.head(self.config.amountImagesTest)

        self.config.logger.log_debug(f"\nTest X: {len(x_img_test_array)}"
                                     f"\nTest Y: {y_df_all_test.shape}")
        self.config.logger.log_info(f"DataFrame Info:\n{y_df_all_test.describe()}\n")

        # Aceita apenas 2 dimensões.
        #x_test = self.reshape_two_dimensions(x_test)

        self.config.logger.log_info(f"\nIniciando predição completa para o R2...\n")

        for index, model in enumerate(self.models):
            # Fazendo a predição sobre os dados de teste
            prediction = model.predict(x_img_test_array)

            # Avaliando com R2
            r2 = r2_score(y_df_all_test, prediction)
            mae = mean_absolute_error(y_df_all_test, prediction)
            mse = mean_squared_error(y_df_all_test, prediction)

            self.config.logger.log_info(f"")
            self.config.logger.log_info(f"====================================================")
            self.config.logger.log_info(f"********** R2 Modelo: {r2} **********")
            self.config.logger.log_info(f"********** MAE [mean]: {mae} **********")
            self.config.logger.log_info(f"********** MSE [mean]: {mse} **********")
            self.config.logger.log_info(f"====================================================")
            self.config.logger.log_info(f"\n")

            if self.config.argsGridSearch > 0:
                self.config.logger.log_info(f"Hiperparâmetros deste modelo:")
                self.config.logger.log_info(f"{self.hyperparameters[index].values}\n")

            self.config.logger.log_info(f"Alguns exemplos de predições ...")
            self.__show_predict_samples(df_amostra, y_df_all_test, prediction)

            del model
        del self.models

    def _min_max_predict_test(self, df_amostra: list, y_df_data: DataFrame, prediction):
        result = []
        for i in tqdm(range(len(df_amostra))):
            amostra = df_amostra[i]
            carbono_predict = prediction[i][0]
            nitrogenio_predict = prediction[i][1]
            carbono_real = y_df_data.iloc[i]['teor_carbono']
            nitrogenio_real = y_df_data.iloc[i]['teor_nitrogenio']
            diff_carbono = abs(carbono_real - carbono_predict)
            diff_nitrogenio = abs(nitrogenio_real - nitrogenio_predict)
            erro_carbono = abs(diff_carbono) / abs(carbono_real) * 100
            erro_nitrogenio = abs(diff_nitrogenio) / abs(nitrogenio_real) * 100

            reg_line = {'amostra': amostra, 'teor_cabono_real': carbono_real, 'teor_cabono_predict': carbono_predict,
                        'teor_cabono_diff': diff_carbono, 'error_carbono(%)': erro_carbono, 'teor_nitrogenio_real': nitrogenio_real,
                        'teor_nitrogenio_predict': nitrogenio_predict, 'teor_nitrogenio_diff': diff_nitrogenio, 'error_nitrogenio(%)': erro_nitrogenio}
            result.append(reg_line)

        df_sorted = pd.DataFrame(result)
        df_sorted = df_sorted.sort_values(by='error_carbono(%)')
        #df_sorted.to_csv('resultado.csv', index=False)
        #self.config.logger.logInfo(f"{df_sorted.to_string(index=False)}")

        df_sorted['grupo'] = df_sorted['amostra'].str.extract(r'([A-Z]+\d+)')[0]

        self.config.logger.log_info(f"\nMelhores resultados ...\n\n")
        self.config.logger.log_info(f"\n{df_sorted.head()}\n")
        self.config.logger.log_info(f"\nPiores resultados ...\n\n")
        self.config.logger.log_info(f"\n{df_sorted.tail()}\n\n")

        df_media_mean_carbono = df_sorted.groupby('grupo').agg(
            {'teor_cabono_predict': 'mean', 'teor_cabono_real': 'first'}).reset_index()
        r2_mean = r2_score(df_media_mean_carbono['teor_cabono_real'], df_media_mean_carbono['teor_cabono_predict'])
        mae_mean = mean_absolute_error(df_media_mean_carbono['teor_cabono_real'], df_media_mean_carbono['teor_cabono_predict'])
        mse_mean = mean_squared_error(df_media_mean_carbono['teor_cabono_real'], df_media_mean_carbono['teor_cabono_predict'])

        self.config.logger.log_info(f"R2 [mean] conjunto de predição CARBONO:")
        self.config.logger.log_info(f"====================================================")
        self.config.logger.log_info(f"====>>>>> R2 [mean]: {r2_mean} <<<<<====")
        self.config.logger.log_info(f"====>>>>> MAE [mean]: {mae_mean} <<<<<====")
        self.config.logger.log_info(f"====>>>>> MSE [mean]: {mse_mean} <<<<<====")
        self.config.logger.log_info(f"====================================================")
        self.config.logger.log_info(f"\n")

        df_media_median_carbono = df_sorted.groupby('grupo').agg(
            {'teor_cabono_predict': 'median', 'teor_cabono_real': 'first'}).reset_index()
        r2_median = r2_score(df_media_median_carbono['teor_cabono_real'], df_media_median_carbono['teor_cabono_predict'])
        mae_median = mean_absolute_error(df_media_median_carbono['teor_cabono_real'], df_media_median_carbono['teor_cabono_predict'])
        mse_median = mean_squared_error(df_media_median_carbono['teor_cabono_real'], df_media_median_carbono['teor_cabono_predict'])

        self.config.logger.log_info(f"R2 [median] conjunto de predição CARBONO:")
        self.config.logger.log_info(f"====================================================")
        self.config.logger.log_info(f"====>>>>> R2 [median]: {r2_median} <<<<<====")
        self.config.logger.log_info(f"====>>>>> MAE [median]: {mae_median} <<<<<====")
        self.config.logger.log_info(f"====>>>>> MSE [median]: {mse_median} <<<<<====")
        self.config.logger.log_info(f"====================================================")
        self.config.logger.log_info(f"\n")

        df_media_mean_nitrogenio = df_sorted.groupby('grupo').agg(
            {'teor_nitrogenio_predict': 'mean', 'teor_nitrogenio_real': 'first'}).reset_index()
        r2_mean = r2_score(df_media_mean_nitrogenio['teor_nitrogenio_real'], df_media_mean_nitrogenio['teor_nitrogenio_predict'])
        mae_mean = mean_absolute_error(df_media_mean_nitrogenio['teor_nitrogenio_real'], df_media_mean_nitrogenio['teor_nitrogenio_predict'])
        mse_mean = mean_squared_error(df_media_mean_nitrogenio['teor_nitrogenio_real'], df_media_mean_nitrogenio['teor_nitrogenio_predict'])

        self.config.logger.log_info(f"R2 [mean] conjunto de predição NITROGENIO:")
        self.config.logger.log_info(f"====================================================")
        self.config.logger.log_info(f"====>>>>> R2 [mean]: {r2_mean} <<<<<====")
        self.config.logger.log_info(f"====>>>>> MAE [mean]: {mae_mean} <<<<<====")
        self.config.logger.log_info(f"====>>>>> MSE [mean]: {mse_mean} <<<<<====")
        self.config.logger.log_info(f"====================================================")
        self.config.logger.log_info(f"\n")

        df_media_median_nitrogenio = df_sorted.groupby('grupo').agg(
            {'teor_nitrogenio_predict': 'median', 'teor_nitrogenio_real': 'first'}).reset_index()
        r2_median = r2_score(df_media_median_nitrogenio['teor_nitrogenio_real'], df_media_median_nitrogenio['teor_nitrogenio_predict'])
        mae_median = mean_absolute_error(df_media_median_nitrogenio['teor_nitrogenio_real'], df_media_median_nitrogenio['teor_nitrogenio_predict'])
        mse_median = mean_squared_error(df_media_median_nitrogenio['teor_nitrogenio_real'], df_media_median_nitrogenio['teor_nitrogenio_predict'])

        self.config.logger.log_info(f"R2 [median] conjunto de predição NITROGENIO:")
        self.config.logger.log_info(f"====================================================")
        self.config.logger.log_info(f"====>>>>> R2 [median]: {r2_median} <<<<<====")
        self.config.logger.log_info(f"====>>>>> MAE [median]: {mae_median} <<<<<====")
        self.config.logger.log_info(f"====>>>>> MSE [median]: {mse_median} <<<<<====")
        self.config.logger.log_info(f"====================================================")
        self.config.logger.log_info(f"\n")