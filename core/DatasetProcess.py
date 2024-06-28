import pandas as pd  # Trabalhar com análise de dados, importação, etc.
import random
from sklearn import preprocessing
from core.ModelConfig import ModelConfig
from core.NormalizeEnum import NormalizeEnum


class DatasetProcess:
    def __init__(self, config: ModelConfig):
        self.config = config

    # Faz o preparado do Dataset para trabalhar no modelo de regressão
    # Retorna dataset limpo, lista de nomes dos arquivos
    @property
    def dataset_process(self):
        # Carregamento do Dataset
        df: pd.DataFrame = pd.read_csv(self.config.pathCSV)

        # Estratégia (1) separando dados de validação.
        # _______________________________________________________________

        # Amostras aleatórias para compor o DataFrame de Validação (apenas no de teste).
        # Essa separação é necessária para ele não misturar as amostras entre os conjuntos
        df_validate = df[df['amostra'].isin(
            ['C2', 'C11', 'C18', 'C28', 'C35', 'C47', 'L3', 'L6', 'L13', 'L16', 'L22', 'L31', 'L39'])]

        # Merge para remover amostras do DataFrame de Validação para o Principal.
        df = (pd.merge(df, df_validate, how='outer', indicator=True)
              .query('_merge == "left_only"')
              .drop('_merge', axis=1))

        # Itens a remover
        # A ideia aqui é se alguma amostra se tornar tão ruim na predição, melhor remover ela do DataFrame
        #itens_remover = ~df['amostra'].isin(['C51', 'L12', 'L5'])
        #df = df[itens_remover]

        # Removendo colunas desnecessárias do DataFrame de Validação
        df_validate = df_validate.drop(
            columns=["class", "qtd_mat_org", "nitrog_calc", "amostra", "classe", "tamanho"])

        # Gerador de Random State
        # A cada treinamento vai embaralhar os dados diferentes
        random_state = random.randint(0, 100)
        self.config.logger.log_info(f"Embaralhamento de dados->random_state: {random_state}")

        # Randomizando DataFrame de Validação
        df_validate = df_validate.sample(frac=1, random_state=random_state, ignore_index=True)

        image_file_names_validate = df_validate["arquivo"].to_list()
        df_validate = df_validate.drop(columns=["arquivo"])
        # _______________________________________________________________

        # Removendo colunas desnecessárias
        df = df.drop(columns=["class", "qtd_mat_org", "nitrog_calc", "amostra", "classe", "tamanho"])

        # Randomizando
        df = df.sample(frac=1, random_state=random_state, ignore_index=True)

        # Separando apenas nomes dos arquivos
        image_file_names = df["arquivo"].to_list()
        # Removendo coluna arquivo para normalização
        df = df.drop(columns=["arquivo"])

        if self.config.argsNormalize == NormalizeEnum.NONE:
            self.config.logger.log_info(f"Informações básicas do Dataset sem normalização ...")
            self.config.logger.log_info(f"DataFrame de dados:\n{df.describe()}\n")
            if not df_validate.empty:
                self.config.logger.log_info(f"DataFrame de validação:\n{df_validate.describe()}\n")
        elif self.config.argsNormalize == NormalizeEnum.Z_Score:
            self.config.logger.log_info(f"Informações básicas do Dataset com normalização Z SCORE ...")
            df_stats = df.describe()
            df_stats = df_stats.transpose()
            df = (df - df_stats['mean']) / df_stats['std']
            self.config.logger.log_info(f"DataFrame de dados:\n{df.describe()}\n")
            if not df_validate.empty:
                df_validate_stats = df_validate.describe()
                df_validate_stats = df_validate_stats.transpose()
                df_validate = (df_validate - df_validate_stats['mean']) / df_validate_stats['std']
                self.config.logger.log_info(f"DataFrame de validação:\n{df_validate.describe()}\n")
        elif self.config.argsNormalize == NormalizeEnum.MinMaxScaler:
            self.config.logger.log_info(f"Informações básicas do Dataset com normalização MinMaxScaler ...")
            scaler = preprocessing.MinMaxScaler()
            df = scaler.fit_transform(df)
            self.config.logger.log_info(f"DataFrame de dados:\n{df.describe()}\n")
            if not df_validate.empty:
                df_validate = scaler.fit_transform(df_validate)
                self.config.logger.log_info(f"DataFrame de validação:\n{df_validate.describe()}\n")
        elif self.config.argsNormalize == NormalizeEnum.RobustScaler:
            self.config.logger.log_info(f"Informações básicas do Dataset com normalização RobustScaler ...")
            scaler = preprocessing.RobustScaler()
            df = scaler.fit_transform(df)
            self.config.logger.log_info(f"DataFrame de dados:\n{df.describe()}\n")
            if not df_validate.empty:
                df_validate = scaler.fit_transform(df_validate)
                self.config.logger.log_info(f"DataFrame de validação:\n{df_validate.describe()}\n")
        elif self.config.argsNormalize == NormalizeEnum.StandardScaler:
            self.config.logger.log_info(f"Informações básicas do Dataset com normalização StandardScaler ...")
            scaler = preprocessing.StandardScaler()
            df = scaler.fit_transform(df)
            self.config.logger.log_info(f"DataFrame de dados:\n{df.describe()}\n")
            if not df_validate.empty:
                df_validate = scaler.fit_transform(df_validate)
                self.config.logger.log_info(f"DataFrame de validação:\n{df_validate.describe()}\n")

        # df = pd.DataFrame(x_scaled, columns=['teor_carbono'])
        # self.config.logger.logInfo(f"{df.describe()}")

        return df, image_file_names, df_validate, image_file_names_validate
