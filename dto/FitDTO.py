from pandas import DataFrame


class FitDTO:
    def __init__(self, x_img_train: list, y_df_train: DataFrame, x_img_validate: list, y_df_validate: DataFrame) -> None:
        self.x_img_train = x_img_train
        self.x_img_validate = x_img_validate
        self.y_df_train = y_df_train
        self.y_df_validate = y_df_validate

    def __str__(self):
        return f"{self.__dict__}"