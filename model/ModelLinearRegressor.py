from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from core.ModelConfig import ModelConfig
from model.ModelABCRegressor import ModelABCRegressor

class ModelLinearRegressor(ModelABCRegressor):
    
    def __init__(self, config : ModelConfig):
        super().__init__(config)
        
    def getSpecialistModel(self, hp):
        return LinearRegression()
    
    def reshapeTwoDimensions(self, X):
        return super().reshapeTwoDimensions(X)
    
    def modelFit(self, models, X_, Y_carbono, X_validate, Y_carbono_validate):
        super().modelFit(models, X_, Y_carbono, X_validate, Y_carbono_validate)
