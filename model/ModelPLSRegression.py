from sklearn.cross_decomposition import PLSRegression
from core.ModelConfig import ModelConfig
from model.ModelABCRegressor import ModelABCRegressor


class ModelPLSRegression(ModelABCRegressor):

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def get_specialist_model(self, hp):
        return PLSRegression(n_components=5, max_iter=1000)

    def reshape_two_dimensions(self, x_data):
        return super().reshape_two_dimensions(x_data)

    def model_fit(self, models, x_data, y_carbono, x_validate, y_carbono_validate):
        super().model_fit(models, x_data, y_carbono, x_validate, y_carbono_validate)
