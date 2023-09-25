
import numpy as np  # Trabalhar com array
import pandas as pd  # Trabalhar com análise de dados, importação, etc.

from entityModelConfig import ModelConfig

# Faz o preparado do Dataset para trabalhar no modelo de regressão
# Retorna dataset limpo, lista de nomes dos arquivos
def dataset_process(modeConfig : ModelConfig):
    # Carregamento do Dataset
    df_train = pd.read_csv(modeConfig.pathCSV)

    # Removendo colunas desnecessárias
    df_train = df_train.drop(
        columns={"class", "qtd_mat_org", "nitrog_calc", "amostra", "classe", "tamanho"})

    # *********************************
    # Excluindo Nitrogênio Por Enquanto
    # *********************************
    df_train = df_train.drop(columns={"teor_nitrogenio"})
    # *********************************

    # Randomizando
    df_train = df_train.sample(frac=1, random_state=1, ignore_index=True)
    
    # Separando apenas nomes dos arquivos
    train_imagefiles = df_train["arquivo"]
    # Removendo coluna arquivo para normalização
    df_train = df_train.drop(columns={"arquivo"})
    
    # Normalização Dataset Treinamento
    train_stats = df_train.describe()
    train_stats = train_stats.transpose()
    df_train = (df_train - train_stats['mean']) / train_stats['std']

    
    return df_train, train_imagefiles