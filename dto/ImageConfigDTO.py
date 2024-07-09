from dto.ModelSetEnum import ModelSetEnum
from shared.infrastructure.log.LoggingPy import LoggingPy


class ImageConfigDTO:
    def __init__(self, logger: LoggingPy = None,  model_set_enum: ModelSetEnum = ModelSetEnum.ResNet50,
                 amount_images_test: int = 3843,  args_preprocess: bool = False) -> None:
        self.logger = logger
        self.modelSetEnum: ModelSetEnum = model_set_enum
        self.amountImagesTest = amount_images_test
        self.argsPreprocess = args_preprocess
        self.imageDimensionX = 256
        self.imageDimensionY = 256
        self.channelColors = 3

    def __str__(self):
        return f"{self.__dict__}"