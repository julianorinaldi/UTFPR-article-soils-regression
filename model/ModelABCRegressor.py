import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from abc import ABC, abstractmethod
from core.ModelConfig import ModelConfig
from core.DatasetProcess import DatasetProcess
from core.ImageProcess import ImageProcess
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from keras_tuner.tuners import Hyperband


class ModelABCRegressor(ABC):
    def __init__(self, config: ModelConfig):
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
    def model_fit(self, models, x_data, y_carbono, x_validate, y_carbono_validate):
        # Juntando os dados de validação com treino no SUPER.
        x_data = np.concatenate((x_data, x_validate), axis=0)
        y_carbono = np.concatenate((y_carbono, y_carbono_validate), axis=0)

        for model in models:
            model.fit(x_data, y_carbono)

    def _show_predict_samples(self, carbono_image_array, img_file_names, cabono_real_array, carbono_prediction_array):
        self._min_max_predict_test(carbono_image_array, img_file_names, cabono_real_array, carbono_prediction_array)

    def _load_images(self, qtd_imagens: int):
        dataset_process = DatasetProcess(self.config)
        df, img_file_names, df_validate, img_file_names_validate = dataset_process.dataset_process

        # Quantidade de imagens usadas para a rede.
        qtd_imagens = len(df)
        if (qtd_imagens > qtd_imagens) and (qtd_imagens > 0):
            qtd_imagens = qtd_imagens

        image_process = ImageProcess(self.config)
        x_validate_train, y_validate_train = np.array([]), np.array([])
        if len(img_file_names_validate) > 0:
            self.config.logger.log_info(f"Carregando imagens ...\n")
            # Array com as imagens a serem carregadas para validação do treino
            image_array_validate = image_process.image_load(img_file_names_validate, qtd_imagens)
            x_validate_train, y_validate_train = image_process.image_convert_array(image_array_validate, df_validate,
                                                                                   qtd_imagens)

        self.config.logger.log_info(f"Carregando imagens ...\n")
        # Array com as imagens a serem carregadas de treino
        image_array = image_process.image_load(img_file_names, qtd_imagens)
        x_train, y_train = image_process.image_convert_array(image_array, df, qtd_imagens)

        # Retorno X_ e y_train, DataFrame, e Lista de Imagens
        # x_validate_train, y_validate_train, df_validate, img_file_names_validate relaciona do Validate do Treino
        return x_train, y_train, x_validate_train, y_validate_train, img_file_names

    def train(self):
        self.config.set_dir_base_img('dataset/images/treinamento-solo-256x256')
        self.config.set_path_csv('dataset/csv/Dataset256x256-Treino.csv')

        x_train, y_train, x_validate_train, y_validate_train, img_file_names = self._load_images(
            qtd_imagens=self.config.amountImagesTrain)

        # Flatten das imagens
        self.config.logger.log_debug(f"Fazendo reshape")

        # Aceita apenas 2 dimensões.
        x_validate_train = self.reshape_two_dimensions(x_validate_train)
        x_train = self.reshape_two_dimensions(x_train)

        self.config.logger.log_debug(f"Novo shape de x_validate_train: {x_validate_train.shape}")
        self.config.logger.log_debug(f"Novo shape de x_train: {x_train.shape}")

        self.config.logger.log_info(f"")
        self.config.logger.log_info(f"Criando modelo: {self.config.modelSetEnum.name}")
        self.config.logger.log_info(f"")

        # Treinar o modelo
        self.config.logger.log_info(f"Iniciando o treino")

        if not self.config.argsGridSearch > 0:
            # Executa sem GridSearch
            self.config.logger.log_info(f"")
            self.config.logger.log_info(f"Executando sem o GridSearch")
            self.config.logger.log_info(f"")
            self.models = {self.get_specialist_model(hp=None)}
            self.model_fit(self.models, x_train, y_train, x_validate_train, y_validate_train)
        else:
            # Executa com GridSearch
            self.config.logger.log_info(f"")
            self.config.logger.log_info(f"Executando com o GridSearch")
            self.config.logger.log_info(f"")
            early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_mae',
                                                              patience=self.config.argsPatience,
                                                              restore_best_weights=True)

            tuner = Hyperband(
                self.get_specialist_model,
                objective='val_mae',
                max_epochs=self.config.argsEpochs,
                factor=3,
                #max_trials=self.config.argsGridSearch,  # Quantas tentativas de hiperparâmetros serão executadas
                #directory='_gridSearchResults',  # diretório para armazenar os resultados
                hyperband_iterations=self.config.argsGridSearch,
                directory='_hyperbandResults',  # diretório para armazenar os resultados
                project_name=self.config.argsNameModel
            )

            if not self.config.argsSepared:
                # Padrão sem separação entre validação e treino      
                x_train = np.concatenate((x_train, x_validate_train), axis=0)
                y_train = np.concatenate((y_train, y_validate_train), axis=0)
                tuner.search(x_train, y_train, epochs=self.config.argsEpochs,
                             validation_split=0.3, callbacks=[early_stopping])
            else:
                # Execute a busca de hiperparâmetros
                tuner.search(x_train, y_train, epochs=self.config.argsEpochs,
                             validation_data=(x_validate_train, y_validate_train),
                             callbacks=[early_stopping])

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

    def test(self):
        # Agora entra o Test
        self.config.set_dir_base_img('dataset/images/teste-solo-256x256')
        self.config.set_path_csv('dataset/csv/Dataset256x256-Teste.csv')

        x_test, y_test, x_validate_test, y_validate_test, img_file_names = self._load_images(
            qtd_imagens=self.config.amountImagesTest)

        # No teste por ignorar estes dados, eles devem estar vazios.
        # x_validate_test, y_validate_test, df_validate, imgFileNamesValidate

        # Aceita apenas 2 dimensões.
        x_test = self.reshape_two_dimensions(x_test)
        y_test = self.reshape_two_dimensions(y_test)

        self.config.logger.log_info(f"")
        self.config.logger.log_info(f"Iniciando predição completa para o R2...")
        self.config.logger.log_info(f"\n")

        for index, model in enumerate(self.models):
            # Fazendo a predição sobre os dados de teste
            prediction = model.predict(x_test)  # type: ignore

            # Avaliando com R2
            r2 = r2_score(y_test, prediction)
            mae = mean_absolute_error(y_test, prediction)
            mse = mean_squared_error(y_test, prediction)

            self.config.logger.log_info(f"")
            self.config.logger.log_info(f"====================================================")
            self.config.logger.log_info(f"********** R2 Modelo: {r2} **********")
            self.config.logger.log_info(f"********** MAE [mean]: {mae} **********")
            self.config.logger.log_info(f"********** MSE [mean]: {mse} **********")
            self.config.logger.log_info(f"====================================================")
            self.config.logger.log_info(f"\n")

            if self.config.argsGridSearch > 0:
                self.config.logger.log_info(f"")
                self.config.logger.log_info(f"Hiperparâmetros deste modelo:")
                self.config.logger.log_info(f"{self.hyperparameters[index].values}")
                self.config.logger.log_info(f"\n")

            self.config.logger.log_info(f"")
            self.config.logger.log_info(f"Alguns exemplos de predições ...")
            self.config.logger.log_info(f"")
            self._show_predict_samples(x_test, img_file_names, y_test, prediction)

            del model
        del self.models

    def _min_max_predict_test(self, carbono_image_array, img_file_names, cabono_real_array, carbono_prediction_array):
        result = []
        for i in tqdm(range(len(cabono_real_array))):
            amostra: str = img_file_names[i]
            predict_value: float = np.array(carbono_prediction_array[i]).item()
            real: float = cabono_real_array[i]
            diff: float = abs(real - predict_value)
            erro: float = abs(diff) / abs(real) * 100

            reg_line = {'amostra': amostra, 'teor_cabono_real': real, 'teor_cabono_predict': predict_value,
                        'teor_cabono_diff': diff, 'error(%)': erro}
            result.append(reg_line)

        df_sorted = pd.DataFrame(result)
        df_sorted = df_sorted.sort_values(by='error(%)')
        #df_sorted.to_csv('resultado.csv', index=False)
        #self.config.logger.logInfo(f"{df_sorted.to_string(index=False)}")

        df_sorted['grupo'] = df_sorted['amostra'].str.extract(r'([A-Z]+\d+)')[0]

        self.config.logger.log_info(f"\nMelhores resultados ...\n\n")
        self.config.logger.log_info(f"\n{df_sorted.head()}\n")
        self.config.logger.log_info(f"\nPiores resultados ...\n\n")
        self.config.logger.log_info(f"\n{df_sorted.tail()}\n\n")

        df_media_mean = df_sorted.groupby('grupo').agg(
            {'teor_cabono_predict': 'mean', 'teor_cabono_real': 'first'}).reset_index()
        r2_mean = r2_score(df_media_mean['teor_cabono_real'], df_media_mean['teor_cabono_predict'])
        mae_mean = mean_absolute_error(df_media_mean['teor_cabono_real'], df_media_mean['teor_cabono_predict'])
        mse_mean = mean_squared_error(df_media_mean['teor_cabono_real'], df_media_mean['teor_cabono_predict'])

        self.config.logger.log_info(f"")
        self.config.logger.log_info(f"R2 [mean] conjunto de predição:")
        self.config.logger.log_info(f"\n")
        self.config.logger.log_info(f"====================================================")
        self.config.logger.log_info(f"====>>>>> R2 [mean]: {r2_mean} <<<<<====")
        self.config.logger.log_info(f"====>>>>> MAE [mean]: {mae_mean} <<<<<====")
        self.config.logger.log_info(f"====>>>>> MSE [mean]: {mse_mean} <<<<<====")
        self.config.logger.log_info(f"====================================================")
        self.config.logger.log_info(f"\n")

        df_media_median = df_sorted.groupby('grupo').agg(
            {'teor_cabono_predict': 'median', 'teor_cabono_real': 'first'}).reset_index()
        r2_median = r2_score(df_media_median['teor_cabono_real'], df_media_median['teor_cabono_predict'])
        mae_median = mean_absolute_error(df_media_median['teor_cabono_real'], df_media_median['teor_cabono_predict'])
        mse_median = mean_squared_error(df_media_median['teor_cabono_real'], df_media_median['teor_cabono_predict'])

        self.config.logger.log_info(f"")
        self.config.logger.log_info(f"R2 [median] conjunto de predição:")
        self.config.logger.log_info(f"\n")
        self.config.logger.log_info(f"====================================================")
        self.config.logger.log_info(f"====>>>>> R2 [median]: {r2_median} <<<<<====")
        self.config.logger.log_info(f"====>>>>> MAE [median]: {mae_median} <<<<<====")
        self.config.logger.log_info(f"====>>>>> MSE [median]: {mse_median} <<<<<====")
        self.config.logger.log_info(f"====================================================")
        self.config.logger.log_info(f"\n")
