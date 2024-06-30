import pandas as pd
import random

from core.NormalizeDataProcess import get_normalize_data
from dto.ConfigModelDTO import ConfigModelDTO


class DatasetProcess:
    def __init__(self, config: ConfigModelDTO):
        self.config = config

    # Carrega o Dataset from CSV
    def __load_dataset_from_csv(self) -> pd.DataFrame:
        # Carregamento do Dataset
        df: pd.DataFrame = pd.read_csv(self.config.pathCSV)
        return df

    # Define amostras aleatórias para compor o DataFrame de Teste (20%)
    def __prepare_dataset_train_and_validate(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        required_columns = {"amostra", "teor_carbono", "teor_nitrogenio", "arquivo"}
        if not required_columns.issubset(df.columns):
            raise Exception(
                "O DataFrame de Treinamento deve conter as colunas 'amostra', 'teor_carbono', 'teor_nitrogenio', 'arquivo'"
            )

        # Define amostras aleatórias para compor o DataFrame de Teste (20%)
        # Essa separação é necessária para ele não misturar as amostras entre os conjuntos
        df_validate = df[df['amostra'].isin(
            ['C2', 'C11', 'C18', 'C28', 'C35', 'C47', 'L3', 'L6', 'L13', 'L16', 'L22', 'L31', 'L39'])]

        # Merge com o DataSet total, para remover amostras que ficaram no Teste.
        # Fica como DataSet Treino
        df_train = (pd.merge(df, df_validate, how='outer', indicator=True)
              .query('_merge == "left_only"')
              .drop('_merge', axis=1))

        # Excluir amostra pois não precisa mais.
        df_train = df_train.drop(columns=["amostra"])
        df_validate = df_validate.drop(columns=["amostra"])

        return df_train, df_validate

    def __prepare_dataset_remove_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=["class", "qtd_mat_org", "nitrog_calc", "classe", "tamanho"])
        return df

    # Gerador de Random State
    # Vai embaralhar os dados de forma diferentes
    # Usar somente depois de separar o dataset em teste e treino
    def __random_dataset(self, df: pd.DataFrame, random_state: int) -> pd.DataFrame:
        # Randomizando DataFrame de Validação
        self.config.logger.log_info(f"Embaralhamento de dados->random_state: {random_state}")
        df_random = df.sample(frac=1, random_state=random_state, ignore_index=True)
        return df_random

    # Faz o preparado do Dataset para trabalhar no modelo de regressão
    # Retorna dataset limpo e separado em treino e teste, e lista dos arquivos imagens
    @property
    def load_and_clean_data(self) -> pd.DataFrame:
        # Carregamento do Dataset
        df_all: pd.DataFrame = self.__load_dataset_from_csv()

        # Removendo colunas desnecessárias
        df_all = self.__prepare_dataset_remove_columns(df_all)

        # Gerador de Random State
        # Vai embaralhar os dados de forma diferentes
        random_state = random.randint(0, 100)
        df_all = self.__random_dataset(df_all, random_state)

        return df_all

    def get_train_validate_process(self, df_all: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        # Normalização dos dados conforme enumerador passado
        df_all = get_normalize_data(df_all, self.config.argsNormalize, self.config.logger)

        # Separa dados de Treinamento e de Test
        df_train, df_validate = self.__prepare_dataset_train_and_validate(df_all)

        return  df_train, df_validate

    def get_test_process(self, df_all: pd.DataFrame) -> pd.DataFrame:
        df_all = df_all.drop(columns=["amostra"])
        # Normalização dos dados conforme enumerador passado
        df_test = get_normalize_data(df_all, self.config.argsNormalize, self.config.logger)

        return df_test