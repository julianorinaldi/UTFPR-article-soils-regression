from sklearn.linear_model import LinearRegression
from entityModelConfig import ModelConfig
from modelABCRegressor import ModelABCRegression

class ModelLinearRegression(ModelABCRegression):
    
    def __init__(self, modelConfig : ModelConfig):
        super().__init__(modelConfig)
        
    def getSpecialistModel(self):
        return LinearRegression()