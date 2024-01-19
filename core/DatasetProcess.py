
import pandas as pd  # Trabalhar com análise de dados, importação, etc.
from sklearn import preprocessing
from core.ModelConfig import ModelConfig

class DatasetProcess:
    def __init__(self, config : ModelConfig):
        self.config = config

    # Faz o preparado do Dataset para trabalhar no modelo de regressão
    # Retorna dataset limpo, lista de nomes dos arquivos
    def dataset_process(self):
        # Carregamento do Dataset
        df : pd.DataFrame = pd.read_csv(self.config.pathCSV)

        # Estratégia (1) separando dados de validação.
        # _______________________________________________________________
    
        # Amostras aleatórias para compor o DataFrame de Validação (apenas no de teste).
        df_validate = df[df['amostra'].isin(['C2', 'C11', 'C18','C28', 'C35','C47', 'L3', 'L6','L13', 'L16', 'L22', 'L31', 'L39'])]
        
        # Merge para remover amostras do DataFrame de Validação para o Principal.
        df = pd.merge(df, df_validate, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
        
        # Itens a remover
        itensRemover = ~df['amostra'].isin(['C51', 'L12', 'L5'])
        df = df[itensRemover]

        # Removendo colunas desnecessárias do DataFrame de Validação
        df_validate = df_validate.drop(columns=["class", "qtd_mat_org", "nitrog_calc", "amostra", "classe", "tamanho", "teor_nitrogenio"])

        # Randomizando DataFrame de Validação
        df_validate = df_validate.sample(frac=1, random_state=1, ignore_index=True)

        imageFileNamesValidate = df_validate["arquivo"].to_list()
        df_validate = df_validate.drop(columns=["arquivo"])
        # _______________________________________________________________
        
        # Removendo colunas desnecessárias
        df = df.drop(columns=["class", "qtd_mat_org", "nitrog_calc", "amostra", "classe", "tamanho"]) 

        # *********************************
        # Excluindo Nitrogênio Por Enquanto
        # *********************************
        df = df.drop(columns=["teor_nitrogenio"]) 
        # *********************************

        # Randomizando
        df = df.sample(frac=1, random_state=1, ignore_index=True)
        
        # Separando apenas nomes dos arquivos
        imageFileNames = df["arquivo"].to_list()
        # Removendo coluna arquivo para normalização
        df = df.drop(columns=["arquivo"])
        
        # Normalização Dataset Treinamento
        # df_stats = df.describe()
        # df_stats = df_stats.transpose()
        # df = (df - df_stats['mean']) / df_stats['std']

        self.config.logger.logInfo(f"Dados do Dataset sem normalização ...")
        self.config.logger.logInfo(f"{df.describe()}")
        if not df_validate.empty:
            self.config.logger.logInfo(f"{df_validate.describe()}")
        

        #x = df.values
        
        # MinMaxScaler
        # self.config.logger.logInfo(f"Normalizando Dataset com MinMaxScaler...")
        # scaler = preprocessing.MinMaxScaler()
        # x_scaled = scaler.fit_transform(x)
        
        #RobustScaler
        # self.config.logger.logInfo(f"Normalizando Dataset com RobustScaler...")
        # scaler = preprocessing.RobustScaler()
        # x_scaled = scaler.fit_transform(x)
        
        # self.config.logger.logInfo(f"Normalizando Dataset com StandardScaler...")
        # scaler = preprocessing.StandardScaler()
        # x_scaled = scaler.fit_transform(x)
        
        # df = pd.DataFrame(x_scaled, columns=['teor_carbono'])
        # self.config.logger.logInfo(f"{df.describe()}")
        
        return df, imageFileNames, df_validate, imageFileNamesValidate