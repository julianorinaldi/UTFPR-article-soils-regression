import pandas as pd

from core.DataProcessBase import DataProcessBase
from core.DataProcessTest import DataProcessTest
from core.NormalizeDataProcess import get_normalize_data
from dto.ConfigTrainModelDTO import ConfigTrainModelDTO


class DataProcessTrain(DataProcessBase):
    PATH_CSV_TRAIN = 'dataset/csv/Dataset256x256-Treino.csv'
    PATH_IMG_TRAIN = 'dataset/images/treinamento-solo-256x256'

    def __init__(self, config: ConfigTrainModelDTO):
        super().__init__(config.logger)
        self.config = config

        # Define amostras aleatórias para compor o DataFrame de Teste (20%)

    def __prepare_dataset_train_and_validate(self, df: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        self.logger.log_debug(f"Preparando dados de treino e validação ...")
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

    # Carrega o Dataset de Treino e Teste.
    # Centraliza aqui para juntar os DataFrames, normalizar e depois separar novamente.
    def load_train_test_data(self) -> (pd.DataFrame, pd.DataFrame):
        df_train = self._load_and_clean_data(DataProcessTrain.PATH_CSV_TRAIN)
        df_test = self._load_and_clean_data(DataProcessTest.PATH_CSV_TEST)

        # Cria uma coluna temporária para identificar origem (treino ou teste)
        df_train['is_train'] = True
        df_test['is_train'] = False

        df_all = pd.concat([df_train, df_test], ignore_index=True)
        df_all = get_normalize_data(df_all, self.config.argsNormalize, self.config.logger)

        # Reconstrói os DataFrames de treino e teste com base nos índices originais
        df_train = df_all[df_all['is_train']].drop(columns=['is_train']).reset_index(drop=True)
        df_test = df_all[~df_all['is_train']].drop(columns=['is_train']).reset_index(drop=True)

        self.logger.log_debug(f"Dados de treino: {len(df_train)} registros\n{df_train.head()}\n\n{df_train.tail()}\n")
        self.logger.log_debug(f"Dados de teste: {len(df_test)} registros\n{df_test.head()}\n\n{df_test.tail()}\n")

        return df_train, df_test

    def get_train_validate_process(self, df_all: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
        # Separa dados de Treinamento e de Test
        df_train, df_validate = self.__prepare_dataset_train_and_validate(df_all)
        return  df_train, df_validate