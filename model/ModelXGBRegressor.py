
from core.ModelConfig import ModelConfig
from model.ModelABCRegressor import ModelABCRegressor
import xgboost as xgb # type: ignore

class ModelXGBRegressor(ModelABCRegressor):

    def __init__(self, config : ModelConfig):
        super().__init__(config)
        
    def getSpecialistModel(self, hp):
        return xgb.XGBRegressor()

    def reshapeTwoDimensions(self, X):
        return super().reshapeTwoDimensions(X)
    
    def modelFit(self, models, X_, Y_carbono, X_validate, Y_carbono_validate):
         super().modelFit(models, X_, Y_carbono, X_validate, Y_carbono_validate)