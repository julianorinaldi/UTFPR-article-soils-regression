from sklearn.linear_model import LinearRegression
from dto.ConfigModelDTO import ConfigModelDTO
from dto.FitDTO import FitDTO
from model.abstract.ModelABCRegressor import ModelABCRegressor


class ModelLinearRegressor(ModelABCRegressor):

    def __init__(self, config: ConfigModelDTO):
        super().__init__(config)

    def get_specialist_model(self, hp):
        return LinearRegression()

    def reshape_two_dimensions(self, x_data):
        return super().reshape_two_dimensions(x_data)

    def model_fit(self, models, fit_dto: FitDTO):
        super().model_fit(models, fit_dto)
