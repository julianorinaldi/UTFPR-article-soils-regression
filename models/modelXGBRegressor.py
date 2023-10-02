
from entityModelConfig import ModelConfig
from models.modelABCRegressor import ModelABCRegressor
import xgboost as xgb

class ModelXGBRegressor(ModelABCRegressor):

    def __init__(self, modelConfig : ModelConfig):
        super().__init__(modelConfig)
        
    def getSpecialistModel(self):
        return xgb.XGBRegressor()
