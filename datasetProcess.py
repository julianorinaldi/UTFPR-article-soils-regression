
import pandas as pd  # Trabalhar com análise de dados, importação, etc.
from sklearn import preprocessing
from entityModelConfig import ModelConfig

# Faz o preparado do Dataset para trabalhar no modelo de regressão
# Retorna dataset limpo, lista de nomes dos arquivos
def dataset_process(modeConfig : ModelConfig):
    # Carregamento do Dataset
    df : pd.DataFrame = pd.read_csv(modeConfig.pathCSV)


    # Estratégia (1) separando dados de validação.
    # _______________________________________________________________
   
    # Amostras aleatórias para compor o DataFrame de Validação (apenas no de teste).
    df_validate = df[df['amostra'].isin(['C2', 'C11', 'C18','C28', 'C35','C47', 'L3', 'L6','L13', 'L16', 'L22', 'L31', 'L39'])]
    
    # Merge para remover amostras do DataFrame de Validação para o Principal.
    df = pd.merge(df, df_validate, how='outer', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    
    # Itens a remover
    #itensRemover = ~df['amostra'].isin(['C51', 'L12', 'L5'])
    #df = df[itensRemover]

    # Removendo colunas desnecessárias do DataFrame de Validação
    df_validate = df_validate.drop(
        columns={"class", "qtd_mat_org", "nitrog_calc", "amostra", "classe", "tamanho", "teor_nitrogenio"}) # type: ignore    

    # Randomizando DataFrame de Validação
    df_validate = df_validate.sample(frac=1, random_state=1, ignore_index=True)

    imageFileNamesValidate = df_validate["arquivo"].to_list()
    df_validate = df_validate.drop(columns={"arquivo"}) # type: ignore
    # _______________________________________________________________
    
    # Removendo colunas desnecessárias
    df = df.drop(
        columns={"class", "qtd_mat_org", "nitrog_calc", "amostra", "classe", "tamanho"}) # type: ignore

    # *********************************
    # Excluindo Nitrogênio Por Enquanto
    # *********************************
    df = df.drop(columns={"teor_nitrogenio"}) # type: ignore
    # *********************************

    # Randomizando
    df = df.sample(frac=1, random_state=1, ignore_index=True)
    
    # Separando apenas nomes dos arquivos
    imageFileNames = df["arquivo"].to_list()
    # Removendo coluna arquivo para normalização
    df = df.drop(columns={"arquivo"}) # type: ignore
    
    # Normalização Dataset Treinamento
    # df_stats = df.describe()
    # df_stats = df_stats.transpose()
    # df = (df - df_stats['mean']) / df_stats['std']

    print(f'{modeConfig.printPrefix} Dados do Dataset sem normalização ...')
    print(f'{df.describe()}')

    x = df.values
    
    # MinMaxScaler
    # print(f'{modeConfig.printPrefix} Normalizando Dataset com MinMaxScaler...')
    # scaler = preprocessing.MinMaxScaler()
    # x_scaled = scaler.fit_transform(x)
    
    #RobustScaler
    # print(f'{modeConfig.printPrefix} Normalizando Dataset com RobustScaler...')
    # scaler = preprocessing.RobustScaler()
    # x_scaled = scaler.fit_transform(x)
    
    # print(f'{modeConfig.printPrefix} Normalizando Dataset com StandardScaler...')
    # scaler = preprocessing.StandardScaler()
    # x_scaled = scaler.fit_transform(x)
    
    # df = pd.DataFrame(x_scaled, columns=['teor_carbono'])
    # print(f'{df.describe()}')
    
    return df, imageFileNames, df_validate, imageFileNamesValidate