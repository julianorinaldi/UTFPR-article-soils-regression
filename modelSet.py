from enum import Enum

def convertModelSet(modelNumber):
    for enumItem in ModelSet:
        if enumItem.value == modelNumber:
            return enumItem
    raise ValueError(f"Nenhum membro de {ModelSet} tem o valor {modelNumber}")

class ModelSet(Enum):
    ResNet50 = 0
    ResNet101 = 1
    ResNet152 = 2
    ConvNeXtBase = 100
    EfficientNetV2S = 300