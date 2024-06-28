from enum import Enum


def convert_model_set_enum(model_number):
    for enumItem in ModelSetEnum:
        if enumItem.value == model_number:
            return enumItem
    raise ValueError(f"Nenhum modelo tem o valor {model_number}")


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

    PLSRegression = 200

    XGBRegressor = 500
    LinearRegression = 510
    SVMLinearRegression = 520
    SVMRBFRegressor = 521
