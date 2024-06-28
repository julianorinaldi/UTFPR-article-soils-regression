from enum import Enum


def convert_normalize_set_enum(normalize):
    for enumItem in NormalizeEnum:
        if enumItem.value == normalize:
            return enumItem
    raise ValueError(f"Nenhum normalizador tem o valor {normalize}")


class NormalizeEnum(Enum):
    NONE = 0
    MinMaxScaler = 1
    RobustScaler = 2
    StandardScaler = 3
    Z_Score = 4
