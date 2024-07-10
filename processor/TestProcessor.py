import numpy as np

from core.ResultLogger import ResultLogger
from dto.ConfigTestDTO import ConfigTestDTO
import pandas as pd
import tensorflow as tf

from core.DataProcessTest import DataProcessTest
from processor.ImageProcessor import ImageProcessor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


class TestProcessor:
    def __init__(self, config: ConfigTestDTO) -> None:
        self.config = config

    def __remove_amostras_ruins(self, df: pd.DataFrame) -> pd.DataFrame:
        #amostras_remove = ['C51']
        amostras_remove = []
        if amostras_remove is None:
            return df

        self.config.logger.log_debug(f"Removendo amostras ruins: {', '.join(map(str, amostras_remove))}")
        df_amostra_remove = df[df['amostra'].isin(amostras_remove)]

        df_result = (pd.merge(df, df_amostra_remove, how='outer', indicator=True)
                    .query('_merge == "left_only"')
                    .drop('_merge', axis=1))
        df_result = df_result.reset_index(drop=True)
        return df_result

    def test(self, df_test: pd.DataFrame, models: list):
        self.config.logger.log_info("Iniciando o Test...\n")

        # Objetiva remover amostrar ruins do conjunto de dados de Teste
        df_test_clean = self.__remove_amostras_ruins(df_test)

        dp = DataProcessTest(self.config)
        df_amostra = df_test_clean["amostra"]
        y_df_all_test = dp.get_test_process(df_all=df_test_clean)

        self.config.logger.log_info(f"Carregando imagens de Teste ...")
        img_p = ImageProcessor(self.config.get_image_config())
        x_img_test_array = img_p.image_load(df=y_df_all_test, path_img=DataProcessTest.PATH_IMG_TEST, amount_img=self.config.amountImagesTest)
        x_img_test_array = tf.stack(x_img_test_array, axis=0)

        # Remover a coluna 'arquivo' para carregar as imagens
        y_df_all_test = y_df_all_test.drop(columns=["arquivo"])
        y_df_all_test = y_df_all_test.head(self.config.amountImagesTest)
        df_amostra = df_amostra.head(self.config.amountImagesTest)

        self.config.logger.log_debug(f"Imagens no Teste no X: {len(x_img_test_array)} - "
                                     f"Shape no Test Y: {y_df_all_test.shape}")

        self.config.logger.log_info(f"Resumo Dados Test:\n{y_df_all_test.describe()}\n")

        # Aceita apenas 2 dimensões.
        #x_test = self.reshape_two_dimensions(x_test)

        self.config.logger.log_info(f"Modelo: {self.config.modelSetEnum}")
        self.config.logger.log_info(f"Iniciando predição completa ...")

        for index, model in enumerate(models):
            # Fazendo a predição sobre os dados de teste
            prediction = model.predict(x_img_test_array)
            prediction = np.round(prediction, 2)
            self.config.logger.log_debug(f"Shape do prediction: {prediction.shape}")

            # Avaliando com R2 Carbono
            r2_c = r2_score(y_df_all_test["teor_carbono"], prediction[:,0])
            mae_c = mean_absolute_error(y_df_all_test["teor_carbono"], prediction[:,0])
            mse_c = mean_squared_error(y_df_all_test["teor_carbono"], prediction[:,0])

            self.config.logger.log_debug(f"Resultado geral de Carbono ...")
            self.config.logger.log_resume(f"")
            self.config.logger.log_resume(f"====================================================")
            self.config.logger.log_resume(f"********** R2 C.: {r2_c} **********")
            self.config.logger.log_resume(f"********** MAE C: {mae_c} **********")
            self.config.logger.log_resume(f"********** MSE C.: {mse_c} **********")
            self.config.logger.log_resume(f"====================================================")
            self.config.logger.log_resume(f"\n\n")

            # Avaliando com R2 Nitrogenio
            r2_n = r2_score(y_df_all_test["teor_nitrogenio"], prediction[:,1])
            mae_n = mean_absolute_error(y_df_all_test["teor_nitrogenio"], prediction[:,1])
            mse_n = mean_squared_error(y_df_all_test["teor_nitrogenio"], prediction[:,1])

            self.config.logger.log_debug(f"Resultado geral de Nitrogênio ...")
            self.config.logger.log_resume(f"")
            self.config.logger.log_resume(f"====================================================")
            self.config.logger.log_resume(f"********** R2 N.: {r2_n} **********")
            self.config.logger.log_resume(f"********** MAE N: {mae_n} **********")
            self.config.logger.log_resume(f"********** MSE N.: {mse_n} **********")
            self.config.logger.log_resume(f"====================================================")
            self.config.logger.log_resume(f"\n\n\n")

            self.config.logger.log_resume(f"Exemplos de predições ...")
            result_logger = ResultLogger(self.config.logger)
            result_logger.logger_predict_test(df_amostra, y_df_all_test, prediction)

            del model
        del models