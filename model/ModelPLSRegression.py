from sklearn.cross_decomposition import PLSRegression
from dto.ConfigTrainModelDTO import ConfigTrainModelDTO
from dto.FitDTO import FitDTO
from model.abstract.ModelABCRegressor import ModelABCRegressor


class ModelPLSRegression(ModelABCRegressor):

    def __init__(self, config: ConfigTrainModelDTO):
        super().__init__(config)

    def get_specialist_model(self, hp):
        return PLSRegression(n_components=5, max_iter=1000)

    def reshape_two_dimensions(self, x_data):
        return super().reshape_two_dimensions(x_data)

    def model_fit(self, models, fit_dto: FitDTO):
        super().model_fit(models, fit_dto)
