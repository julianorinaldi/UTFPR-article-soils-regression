from enum import Enum

def convertModelSetEnum(modelNumber):
    for enumItem in ModelSetEnum:
        if enumItem.value == modelNumber:
            return enumItem
    raise ValueError(f"Nenhum modelo tem o valor {modelNumber}")

class ModelSetEnum(Enum):
    ResNet50 = 0
    ResNet101 = 1
    ResNet152 = 2
    
    ConvNeXtBase = 10
    ConvNeXtXLarge = 11
    
    EfficientNetB7 = 20
    EfficientNetV2S = 21
    EfficientNetV2L = 22
    
    InceptionResNetV2 = 30
    
    DenseNet169 = 40
    
    VGG19 = 50
    CNN = 100
    
    XGBRegressor = 500
    LinearRegression = 510
    SVMLinearRegression = 520
    SVMRBFRegressor = 521