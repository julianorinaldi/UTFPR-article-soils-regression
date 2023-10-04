from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from entityModelConfig import ModelConfig
from models.modelABCRegressor import ModelABCRegressor

class ModelLinearRegressor(ModelABCRegressor):
    
    def __init__(self, modelConfig : ModelConfig):
        super().__init__(modelConfig)
        
    def getSpecialistModel(self):
        # Força que o modelo seja linear positivo
        return LinearRegression(positive=True)
    
    def reshapeTwoDimensions(self, X):
        return super().reshapeTwoDimensions(X)
    
    def modelFit(self, model, X_, Y_carbono):
        rfe = RFE(model, n_features_to_select=500)
        _modelFit = rfe.fit(X_, Y_carbono)
        return _modelFit