from entityModelConfig import ModelConfig
from models.modelABCRegressor import ModelABCRegressor
from sklearn.svm import SVR

class ModelSVMRBFRegressor(ModelABCRegressor):

    def __init__(self, modelConfig : ModelConfig):
        super().__init__(modelConfig)
        
    def getSpecialistModel(self):
        return SVR(kernel='rbf')

    def reshapeTwoDimensions(self, X):
        return super().reshapeTwoDimensions(X)
    
    def modelFit(self, model, X_, Y_carbono, X_validate, Y_carbono_validate):
        super().modelFit(model, X_, Y_carbono, X_validate, Y_carbono_validate)
        
        