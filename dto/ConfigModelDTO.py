from dto.ModelSetEnum import ModelSetEnum
from dto.NormalizeEnum import NormalizeEnum
from shared.infrastructure.log.LoggingPy import LoggingPy


class ConfigModelDTO:
    PATH_CSV_TRAIN = 'dataset/csv/Dataset256x256-Treino.csv'
    PATH_IMG_TRAIN = 'dataset/images/treinamento-solo-256x256'
    PATH_CSV_TEST = 'dataset/csv/Dataset256x256-Teste.csv'
    PATH_IMG_TEST = 'dataset/images/teste-solo-256x256'

    def __init__(self, logger: LoggingPy = None,  model_set_enum: ModelSetEnum = ModelSetEnum.ResNet50,
                 log_level: int = 1, image_dimension_x: int = 256, image_dimension_y: int = 256, channel_colors: int = 3,
                 amount_images_train: int = 8930, amount_images_test: int = 3843,
                 args_name_model: str = '', args_normalize: NormalizeEnum = NormalizeEnum.NONE, args_trainable: bool = False,
                 args_separed: bool = False, args_preprocess: bool = False, args_only_test: bool = False,
                 args_epochs: int = 100, args_patience: int = 5, args_grid_search: int = 0,
                 args_show_model: bool = False) -> None:
        self.logger = logger
        self.modelSetEnum: ModelSetEnum = model_set_enum
        self.log_level = log_level
        self.imageDimensionX = image_dimension_x
        self.imageDimensionY = image_dimension_y
        self.channelColors = channel_colors
        self.amountImagesTrain = amount_images_train
        self.amountImagesTest = amount_images_test
        self.argsNameModel = args_name_model
        self.argsNormalize: NormalizeEnum = args_normalize
        self.argsTrainable = args_trainable
        self.argsSepared = args_separed
        self.argsPreprocess = args_preprocess
        self.argsOnlyTest = args_only_test
        self.argsEpochs = args_epochs
        self.argsPatience = args_patience
        self.argsGridSearch = args_grid_search
        self.argsShowModel = args_show_model

    def __str__(self):
        return f"{self.__dict__}"
