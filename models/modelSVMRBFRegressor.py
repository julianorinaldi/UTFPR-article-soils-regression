from entityModelConfig import ModelConfig
from models.modelABCRegressor import ModelABCRegressor
from sklearn.svm import SVR

class ModelSVMRBFRegressor(ModelABCRegressor):

    def __init__(self, modelConfig : ModelConfig):
        super().__init__(modelConfig)
        
    def getSpecialistModel(self, hp):
        return SVR(kernel='rbf')

    def reshapeTwoDimensions(self, X):
        return super().reshapeTwoDimensions(X)
    
    def modelFit(self, models, X_, Y_carbono, X_validate, Y_carbono_validate):
        super().modelFit(models, X_, Y_carbono, X_validate, Y_carbono_validate)
        
        