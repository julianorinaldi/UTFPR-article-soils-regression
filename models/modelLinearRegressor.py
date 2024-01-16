from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from entityModelConfig import ModelConfig
from models.modelABCRegressor import ModelABCRegressor

class ModelLinearRegressor(ModelABCRegressor):
    
    def __init__(self, modelConfig : ModelConfig):
        super().__init__(modelConfig)
        
    def getSpecialistModel(self):
        return LinearRegression()
    
    def reshapeTwoDimensions(self, X):
        return super().reshapeTwoDimensions(X)
    
    def modelFit(self, model, X_, Y_carbono, X_validate, Y_carbono_validate):
        super().modelFit(model, X_, Y_carbono, X_validate, Y_carbono_validate)
