from enum import Enum

def convertModelSet(modelNumber):
    for enumItem in ModelSet:
        if enumItem.value == modelNumber:
            return enumItem
    raise ValueError(f"Nenhum modelo tem o valor {modelNumber}")

class ModelSet(Enum):
    ResNet50 = 0
    ResNet101 = 1
    ResNet152 = 2
    ConvNeXtBase = 3
    EfficientNetB7 = 4
    EfficientNetV2S = 5
    InceptionResNetV2 = 6
    DenseNet169 = 7
    VGG19 = 8
    RandomForestRegressor = 100