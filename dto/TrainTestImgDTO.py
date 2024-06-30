from pandas import DataFrame


class TrainTestImgDTO:
    def __init__(self, df_train: DataFrame, df_test: DataFrame, img_list_train: list, img_list_test: list) -> None:
        self.df_train: DataFrame = df_train
        self.df_test: DataFrame = df_test
        self.img_list_train: list = img_list_train
        self.img_list_test: list = img_list_test

    def __str__(self):
        return f"{self.__dict__}"