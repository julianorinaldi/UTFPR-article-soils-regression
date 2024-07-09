from dto.ImageConfigDTO import ImageConfigDTO
from dto.ModelSetEnum import ModelSetEnum
from shared.infrastructure.log.LoggingPy import LoggingPy


class ConfigTestDTO:

    def __init__(self, logger: LoggingPy = None,  model_set_enum: ModelSetEnum = ModelSetEnum.ResNet50,
                 log_level: int = 1, amount_images_test: int = 3843,
                 args_name_model: str = '',  args_preprocess: bool = False,
                 args_show_model: bool = False) -> None:
        self.logger = logger
        self.modelSetEnum: ModelSetEnum = model_set_enum
        self.log_level = log_level
        self.amountImagesTest = amount_images_test
        self.argsNameModel = args_name_model
        self.argsPreprocess = args_preprocess
        self.argsShowModel = args_show_model

    def get_image_config(self) -> ImageConfigDTO:
        return ImageConfigDTO(logger=self.logger, model_set_enum=self.modelSetEnum,
                              amount_images_test=self.amountImagesTest, args_preprocess=self.argsPreprocess)
    def __str__(self):
        return f"{self.__dict__}"
