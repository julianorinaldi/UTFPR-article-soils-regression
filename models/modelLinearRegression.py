from sklearn.linear_model import LinearRegression
from modelABCRegressor import ModelABCRegression

class ModelLinearRegression(ModelABCRegression):
    
    def getSpecialistModel(self):
        return LinearRegression()