from shared.infrastructure.log.LoggingPy import LoggingPy
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class ResultLogger:

    def __init__(self, logger: LoggingPy):
        self.logger = logger

    def logger_predict_test(self, df_amostra: list, y_df_data: DataFrame, prediction):
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
                        'teor_cabono_diff': diff_carbono, 'error_carbono(%)': erro_carbono,
                        'teor_nitrogenio_real': nitrogenio_real,
                        'teor_nitrogenio_predict': nitrogenio_predict, 'teor_nitrogenio_diff': diff_nitrogenio,
                        'error_nitrogenio(%)': erro_nitrogenio}
            result.append(reg_line)

        df_sorted = pd.DataFrame(result)
        df_sorted['grupo'] = df_sorted['amostra'].str.extract(r'([A-Z]+\d+)')[0]
        df_sorted_carbono = df_sorted.sort_values(by='error_carbono(%)')
        # df_sorted.to_csv('resultado.csv', index=False)
        #  self.logger.logInfo(f"{df_sorted.to_string(index=False)}")
        columns_remove_nitrogenio = ["teor_nitrogenio_real", "teor_nitrogenio_predict", "teor_nitrogenio_diff",
                                     "error_nitrogenio(%)"]
        df_sorted_carbono = df_sorted_carbono.drop(columns=columns_remove_nitrogenio)
        self.logger.log_resume(f"Melhores resultados Carbono ...\n{df_sorted_carbono.head()}\n")
        self.logger.log_resume(f"Piores resultados Carbono ...\n{df_sorted_carbono.tail()}\n")


        df_sorted_nitrogenio = df_sorted.sort_values(by='error_nitrogenio(%)')
        columns_remove_carbono = ["teor_cabono_real", "teor_cabono_predict", "teor_cabono_diff", "error_carbono(%)"]
        df_sorted_nitrogenio = df_sorted_nitrogenio.drop(columns=columns_remove_carbono)
        self.logger.log_resume(f"Melhores resultados Nitrogênio ...\n{df_sorted_nitrogenio.head()}\n")
        self.logger.log_resume(f"Piores resultados Nitrogênio ...\n{df_sorted_nitrogenio.tail()}\n\n")

        df_media_mean_carbono = df_sorted.groupby('grupo').agg(
            {'teor_cabono_predict': 'mean', 'teor_cabono_real': 'first'}).reset_index()
        df_media_mean_carbono['error_carbono(%)'] = (
                abs(df_media_mean_carbono['teor_cabono_real'] - df_media_mean_carbono["teor_cabono_predict"]) /
                    abs(df_media_mean_carbono['teor_cabono_real']))
        df_media_mean_carbono = df_media_mean_carbono.sort_values(by='error_carbono(%)')

        r2_mean = r2_score(df_media_mean_carbono['teor_cabono_real'], df_media_mean_carbono['teor_cabono_predict'])
        mae_mean = mean_absolute_error(df_media_mean_carbono['teor_cabono_real'],
                                       df_media_mean_carbono['teor_cabono_predict'])
        mse_mean = mean_squared_error(df_media_mean_carbono['teor_cabono_real'],
                                      df_media_mean_carbono['teor_cabono_predict'])

        self.logger.log_resume(f"R2 [mean] conjunto de predição CARBONO:")
        self.logger.log_resume(f"====================================================")
        self.logger.log_resume(f"====>>>>> R2 [mean]: {r2_mean} <<<<<====")
        self.logger.log_resume(f"====>>>>> MAE [mean]: {mae_mean} <<<<<====")
        self.logger.log_resume(f"====>>>>> MSE [mean]: {mse_mean} <<<<<====")
        self.logger.log_resume(f"====================================================")
        self.logger.log_resume(f"\n")

        self.logger.log_resume(f"Melhores resultados CARBONO [mean]...\n{df_media_mean_carbono.head()}\n")
        self.logger.log_resume(f"Piores resultados CARBONO [mean]...\n{df_media_mean_carbono.tail()}\n\n")

        df_media_median_carbono = df_sorted.groupby('grupo').agg(
            {'teor_cabono_predict': 'median', 'teor_cabono_real': 'first'}).reset_index()
        df_media_median_carbono['error_carbono(%)'] = (
                abs(df_media_median_carbono['teor_cabono_real'] - df_media_median_carbono["teor_cabono_predict"]) /
                    abs(df_media_median_carbono['teor_cabono_real']))
        df_media_median_carbono = df_media_median_carbono.sort_values(by='error_carbono(%)')

        r2_median = r2_score(df_media_median_carbono['teor_cabono_real'],
                             df_media_median_carbono['teor_cabono_predict'])
        mae_median = mean_absolute_error(df_media_median_carbono['teor_cabono_real'],
                                         df_media_median_carbono['teor_cabono_predict'])
        mse_median = mean_squared_error(df_media_median_carbono['teor_cabono_real'],
                                        df_media_median_carbono['teor_cabono_predict'])

        self.logger.log_resume(f"R2 [median] conjunto de predição CARBONO:")
        self.logger.log_resume(f"====================================================")
        self.logger.log_resume(f"====>>>>> R2 [median]: {r2_median} <<<<<====")
        self.logger.log_resume(f"====>>>>> MAE [median]: {mae_median} <<<<<====")
        self.logger.log_resume(f"====>>>>> MSE [median]: {mse_median} <<<<<====")
        self.logger.log_resume(f"====================================================")
        self.logger.log_resume(f"\n")

        self.logger.log_resume(f"Melhores resultados CARBONO [median]...\n{df_media_median_carbono.head()}\n")
        self.logger.log_resume(f"Piores resultados CARBONO [median]...\n{df_media_median_carbono.tail()}\n\n")


        df_media_mean_nitrogenio = df_sorted.groupby('grupo').agg(
            {'teor_nitrogenio_predict': 'mean', 'teor_nitrogenio_real': 'first'}).reset_index()

        df_media_mean_nitrogenio['error_nitrogenio(%)'] = (
                abs(df_media_mean_nitrogenio['teor_nitrogenio_real'] - df_media_mean_nitrogenio["teor_nitrogenio_predict"]) /
                    abs(df_media_mean_nitrogenio['teor_nitrogenio_real']))
        df_media_mean_nitrogenio = df_media_mean_nitrogenio.sort_values(by='error_nitrogenio(%)')

        r2_mean = r2_score(df_media_mean_nitrogenio['teor_nitrogenio_real'],
                           df_media_mean_nitrogenio['teor_nitrogenio_predict'])
        mae_mean = mean_absolute_error(df_media_mean_nitrogenio['teor_nitrogenio_real'],
                                       df_media_mean_nitrogenio['teor_nitrogenio_predict'])
        mse_mean = mean_squared_error(df_media_mean_nitrogenio['teor_nitrogenio_real'],
                                      df_media_mean_nitrogenio['teor_nitrogenio_predict'])

        self.logger.log_resume(f"R2 [mean] conjunto de predição NITROGENIO:")
        self.logger.log_resume(f"====================================================")
        self.logger.log_resume(f"====>>>>> R2 [mean]: {r2_mean} <<<<<====")
        self.logger.log_resume(f"====>>>>> MAE [mean]: {mae_mean} <<<<<====")
        self.logger.log_resume(f"====>>>>> MSE [mean]: {mse_mean} <<<<<====")
        self.logger.log_resume(f"====================================================")
        self.logger.log_resume(f"\n")

        self.logger.log_resume(f"Melhores resultados NITROGENIO [mean]...\n{df_media_mean_nitrogenio.head()}\n")
        self.logger.log_resume(f"Piores resultados NITROGENIO [mean]...\n{df_media_mean_nitrogenio.tail()}\n\n")

        df_media_median_nitrogenio = df_sorted.groupby('grupo').agg(
            {'teor_nitrogenio_predict': 'median', 'teor_nitrogenio_real': 'first'}).reset_index()
        df_media_median_nitrogenio['error_nitrogenio(%)'] = (
                abs(df_media_median_nitrogenio['teor_nitrogenio_real'] - df_media_median_nitrogenio["teor_nitrogenio_predict"]) /
                    abs(df_media_median_nitrogenio['teor_nitrogenio_real']))
        df_media_median_nitrogenio = df_media_median_nitrogenio.sort_values(by='error_nitrogenio(%)')

        r2_median = r2_score(df_media_median_nitrogenio['teor_nitrogenio_real'],
                             df_media_median_nitrogenio['teor_nitrogenio_predict'])
        mae_median = mean_absolute_error(df_media_median_nitrogenio['teor_nitrogenio_real'],
                                         df_media_median_nitrogenio['teor_nitrogenio_predict'])
        mse_median = mean_squared_error(df_media_median_nitrogenio['teor_nitrogenio_real'],
                                        df_media_median_nitrogenio['teor_nitrogenio_predict'])

        self.logger.log_resume(f"R2 [median] conjunto de predição NITROGENIO:")
        self.logger.log_resume(f"====================================================")
        self.logger.log_resume(f"====>>>>> R2 [median]: {r2_median} <<<<<====")
        self.logger.log_resume(f"====>>>>> MAE [median]: {mae_median} <<<<<====")
        self.logger.log_resume(f"====>>>>> MSE [median]: {mse_median} <<<<<====")
        self.logger.log_resume(f"====================================================")
        self.logger.log_resume(f"\n")

        self.logger.log_resume(f"Melhores resultados NITROGENIO [median]...\n{df_media_median_nitrogenio.head()}\n")
        self.logger.log_resume(f"Piores resultados NITROGENIO [median]...\n{df_media_median_nitrogenio.tail()}\n\n")