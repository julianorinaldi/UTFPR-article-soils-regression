from modelSet import ModelSet

class ModelConfig:
    def __init__(self, modelSet : ModelSet = ModelSet.ResNet50, 
                pathCSV : str = 'dataset/csv/Dataset256x256-Treino.csv', 
                dir_base_img : str = 'dataset/images/treinamento-solo-256x256', 
                imageDimensionX : int = 256, imageDimensionY : int = 256, channelColors : int = 3,
                amountImagesTrain : int = 8930, amountImagesTest : int = 3843,
                argsNameModel : str = '', argsDebug : bool = False, argsTrainable : bool = False,
                argsSepared : bool = False, argsPreprocess : bool = False, argsOnlyTest : bool = False, 
                argsEpochs : int = 100, argsPatience : int = 5, argsGridSearch : bool = False, 
                printPrefix : str = '>>>>>>>>>>>>>>>>>'):
        self.modelSet = modelSet
        self.imageDimensionX = imageDimensionX
        self.imageDimensionY = imageDimensionY
        self.channelColors = channelColors
        self.amountImagesTrain = amountImagesTrain
        self.amountImagesTest = amountImagesTest
        self.dirBaseImg = dir_base_img
        self.pathCSV = pathCSV
        self.argsNameModel = argsNameModel
        self.argsDebug = argsDebug
        self.argsTrainable = argsTrainable
        self.argsSepared = argsSepared
        self.argsPreprocess = argsPreprocess
        self.argsOnlyTest = argsOnlyTest
        self.argsEpochs = argsEpochs
        self.argsPatience = argsPatience
        self.argsGridSearch = argsGridSearch
        self.printPrefix = printPrefix
    
    def setPathCSV(self, pathCSV):
        self.pathCSV = pathCSV
        
    def setDirBaseImg(self, dir_base_img):
        self.dirBaseImg = dir_base_img
        
    def __str__(self):
        return f"{self.__dict__}"