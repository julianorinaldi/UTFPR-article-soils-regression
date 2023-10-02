from sklearn.linear_model import LinearRegression
from entityModelConfig import ModelConfig
from models.modelABCRegressor import ModelABCRegressor

class ModelLinearRegressor(ModelABCRegressor):
    
    def __init__(self, modelConfig : ModelConfig):
        super().__init__(modelConfig)
        
    def getSpecialistModel(self):
        return LinearRegression()