from pandas import DataFrame


class TrainTestArrayDTO:
    def __init__(self, df_train: DataFrame, df_validation: DataFrame, img_list_train: list, img_list_validation: list) -> None:
        self.df_train: DataFrame = df_train
        self.df_validation: DataFrame = df_validation
        self.img_list_train: list = img_list_train
        self.img_list_validation: list = img_list_validation

    def __str__(self):
        return f"{self.__dict__}"