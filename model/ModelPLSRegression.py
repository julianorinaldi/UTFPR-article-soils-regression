from sklearn.cross_decomposition import PLSRegression
from core.ModelConfig import ModelConfig
from model.ModelABCRegressor import ModelABCRegressor

class ModelPLSRegression(ModelABCRegressor):
    
    def __init__(self, config : ModelConfig):
        super().__init__(config)
        
    def getSpecialistModel(self, hp):
        return PLSRegression(n_components=5, max_iter=1000)
    
    def reshapeTwoDimensions(self, X):
        return super().reshapeTwoDimensions(X)
    
    def modelFit(self, models, X_, Y_carbono, X_validate, Y_carbono_validate):
        super().modelFit(models, X_, Y_carbono, X_validate, Y_carbono_validate)
