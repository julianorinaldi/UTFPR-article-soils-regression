import pandas as pd
import random

from shared.infrastructure.log.LoggingPy import LoggingPy

class DataProcessBase:
    def __init__(self, logger: LoggingPy):
        self.logger: LoggingPy = logger

    def __prepare_dataset_remove_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        columns_remove = ["class", "qtd_mat_org", "nitrog_calc", "classe", "tamanho"]
        self.logger.log_debug(f"Removendo colunas: {', '.join(map(str, columns_remove))}")
        df = df.drop(columns=columns_remove)
        return df

        # Gerador de Random State
        # Vai embaralhar os dados de forma diferentes
        # Usar somente depois de separar o dataset em teste e treino

    def __random_dataset(self, df: pd.DataFrame, random_state: int) -> pd.DataFrame:
        # Randomizando DataFrame de Validação
        self.logger.log_info(f"Embaralhamento de dados->random_state: {random_state}")
        df_random = df.sample(frac=1, random_state=random_state, ignore_index=True)
        return df_random

    def _load_dataset_from_csv(self, path_csv: str) -> pd.DataFrame:
        self.logger.log_debug(f"Carregando dataset: {path_csv}")
        df: pd.DataFrame = pd.read_csv(path_csv)
        return df

    # Faz o preparado do Dataset para trabalhar no modelo de regressão
    # Retorna dataset limpo e separado em treino e teste, e lista dos arquivos imagens
    def _load_and_clean_data(self, path_csv: str, random_state: int = 0) -> pd.DataFrame:
        # Carregamento do Dataset
        df_all: pd.DataFrame = self._load_dataset_from_csv(path_csv)

        # Removendo colunas desnecessárias
        df_all = self.__prepare_dataset_remove_columns(df_all)

        if random_state == 0:
            # Gerador de Random State
            # Vai embaralhar os dados de forma diferentes
            random_state = random.randint(0, 100)

        df_all = self.__random_dataset(df_all, random_state)

        return df_all