
import pandas as pd  # Trabalhar com análise de dados, importação, etc.
from sklearn import preprocessing
from entityModelConfig import ModelConfig

# Faz o preparado do Dataset para trabalhar no modelo de regressão
# Retorna dataset limpo, lista de nomes dos arquivos
def dataset_process(modeConfig : ModelConfig):
    # Carregamento do Dataset
    df = pd.read_csv(modeConfig.pathCSV)

    # Removendo colunas desnecessárias
    df = df.drop(
        columns={"class", "qtd_mat_org", "nitrog_calc", "amostra", "classe", "tamanho"})

    # *********************************
    # Excluindo Nitrogênio Por Enquanto
    # *********************************
    df = df.drop(columns={"teor_nitrogenio"})
    # *********************************

    # Randomizando
    df = df.sample(frac=1, random_state=1, ignore_index=True)
    
    # Separando apenas nomes dos arquivos
    imagefiles = df["arquivo"].to_list()
    # Removendo coluna arquivo para normalização
    df = df.drop(columns={"arquivo"})
    
    # Normalização Dataset Treinamento
    # df_stats = df.describe()
    # df_stats = df_stats.transpose()
    # df = (df - df_stats['mean']) / df_stats['std']

    # Normalização MinMaxScaler
    print(f'{df.head()}')
    print(f'{modeConfig.printPrefix} Normalizando Dataset...')
    scaler = preprocessing.MinMaxScaler()
    df = scaler.fit_transform(df)
    print(f'{df.head()}')
    
    return df, imagefiles