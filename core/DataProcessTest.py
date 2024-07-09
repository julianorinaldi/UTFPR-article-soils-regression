from core.DataProcessBase import DataProcessBase
from dto.ConfigTestDTO import ConfigTestDTO
import pandas as pd

class DataProcessTest(DataProcessBase):
    PATH_CSV_TEST = 'dataset/csv/Dataset256x256-Teste.csv'
    PATH_IMG_TEST = 'dataset/images/teste-solo-256x256'

    def __init__(self, config: ConfigTestDTO):
        super().__init__(config.logger)
        self.config = config

    def get_test_process(self, df_all: pd.DataFrame) -> pd.DataFrame:
        columns_remove = ["amostra"]
        self.logger.log_debug(f"Removendo colunas: {', '.join(map(str, columns_remove))}")
        df_test = df_all.drop(columns=columns_remove)
        return df_test

    # Carrega o Dataset de Treino e Teste.
    # Centraliza aqui para juntar os DataFrames, normalizar e depois separar novamente.
    def load_test_data(self) -> pd.DataFrame:
        df_test = self._load_and_clean_data(DataProcessTest.PATH_CSV_TEST)
        self.logger.log_debug(f"Dados de teste: {len(df_test)} registros\n{df_test.head()}\n{df_test.tail()}")
        return df_test