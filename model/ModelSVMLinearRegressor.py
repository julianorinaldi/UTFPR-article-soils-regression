from core.ModelConfig import ModelConfig
from model.ModelABCRegressor import ModelABCRegressor
from sklearn.svm import SVR

class ModelSVMLinearRegressor(ModelABCRegressor):

    def __init__(self, config : ModelConfig):
        super().__init__(config)
        
    def getSpecialistModel(self, hp):
        return SVR(kernel='linear')

    def reshapeTwoDimensions(self, X):
        return super().reshapeTwoDimensions(X)
    
    def modelFit(self, models, X_, Y_carbono, X_validate, Y_carbono_validate):
         super().modelFit(models, X_, Y_carbono, X_validate, Y_carbono_validate)