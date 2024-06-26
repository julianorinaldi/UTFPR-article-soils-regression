from pandas import DataFrame
from sklearn import preprocessing

from dto.NormalizeEnum import NormalizeEnum


def __normalize_z_score(df: DataFrame):
    df_stats = df.describe().transpose()
    return (df - df_stats['mean']) / df_stats['std']

def __normalize_other(df: DataFrame, scaler_class):
    scaler = scaler_class()
    return scaler.fit_transform(df)

__normalization_functions = {
    NormalizeEnum.NONE: lambda df: df,  # Não faz nada
    NormalizeEnum.Z_Score: __normalize_z_score,
    NormalizeEnum.MinMaxScaler: lambda df: __normalize_other(df, preprocessing.MinMaxScaler),
    NormalizeEnum.RobustScaler: lambda df: __normalize_other(df, preprocessing.RobustScaler),
    NormalizeEnum.StandardScaler: lambda df: __normalize_other(df, preprocessing.StandardScaler)
}

def get_normalize_data(df: DataFrame, normalize: NormalizeEnum, logger) -> DataFrame:
    logger.log_info(f"Informações básicas do Dataset com normalização {normalize.name} ...")
    normalize_func = __normalization_functions.get(normalize, lambda df_inner: df_inner)
    numeric_columns = df.select_dtypes(include='number')  # Seleciona apenas colunas numéricas
    df_normalized = normalize_func(numeric_columns)
    df[numeric_columns.columns] = df_normalized  # Substitui as colunas originais pelas normalizadas
    return df