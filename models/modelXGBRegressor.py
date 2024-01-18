
from entityModelConfig import ModelConfig
from models.modelABCRegressor import ModelABCRegressor
import xgboost as xgb

class ModelXGBRegressor(ModelABCRegressor):

    def __init__(self, modelConfig : ModelConfig):
        super().__init__(modelConfig)
        
    def getSpecialistModel(self, hp):
        return xgb.XGBRegressor()

    def reshapeTwoDimensions(self, X):
        return super().reshapeTwoDimensions(X)
    
    def modelFit(self, models, X_, Y_carbono, X_validate, Y_carbono_validate):
         super().modelFit(models, X_, Y_carbono, X_validate, Y_carbono_validate)