from core.ModelSetEnum import ModelSetEnum
from log.LoggingPy import LoggingPy

class ModelConfig:
    def __init__(self, modelSetEnum : ModelSetEnum = ModelSetEnum.ResNet50, 
                log_level : int = 1, pathCSV : str = 'dataset/csv/Dataset256x256-Treino.csv', 
                dir_base_img : str = 'dataset/images/treinamento-solo-256x256', 
                imageDimensionX : int = 256, imageDimensionY : int = 256, channelColors : int = 3,
                amountImagesTrain : int = 8930, amountImagesTest : int = 3843,
                argsNameModel : str = '', argsTrainable : bool = False,
                argsSepared : bool = False, argsPreprocess : bool = False, argsOnlyTest : bool = False, 
                argsEpochs : int = 100, argsPatience : int = 5, argsGridSearch : int = 0, 
                argsShowModel : bool = False) -> None:
        self.modelSetEnum = modelSetEnum
        self.logger : LoggingPy
        self.log_level = log_level
        self.imageDimensionX = imageDimensionX
        self.imageDimensionY = imageDimensionY
        self.channelColors = channelColors
        self.amountImagesTrain = amountImagesTrain
        self.amountImagesTest = amountImagesTest
        self.dirBaseImg = dir_base_img
        self.pathCSV = pathCSV
        self.argsNameModel = argsNameModel
        self.argsTrainable = argsTrainable
        self.argsSepared = argsSepared
        self.argsPreprocess = argsPreprocess
        self.argsOnlyTest = argsOnlyTest
        self.argsEpochs = argsEpochs
        self.argsPatience = argsPatience
        self.argsGridSearch = argsGridSearch
        self.argsShowModel = argsShowModel

    def setLogger(self, logger : LoggingPy):
        self.logger = logger
    
    def setPathCSV(self, pathCSV):
        self.pathCSV = pathCSV
        
    def setDirBaseImg(self, dir_base_img):
        self.dirBaseImg = dir_base_img
        
    def __str__(self):
        return f"{self.__dict__}"