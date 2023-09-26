from modelSet import ModelSet

class ModelConfig:
    def __init__(self, modelSet : ModelSet = ModelSet.ResNet50, 
                pathCSV : str = 'dataset/csv/Dataset256x256-Treino.csv', 
                dir_base_img : str = 'dataset/images/treinamento-solo-256x256', 
                imageDimensionX : int = 256, imageDimensionY : int = 256, channelColors : int = 3,
                argsNameModel : str = '', argsDebug : bool = False, argsTrainable : bool = False,
                argsPreprocess : bool = False, printPrefix : str = '>>>>>>>>>>>>>>>>>'):
        self.modelSet = modelSet
        self.imageDimensionX = imageDimensionX
        self.imageDimensionY = imageDimensionY
        self.channelColors = channelColors
        self.setDirBaseImg(dir_base_img)
        self.setPathCSV(pathCSV)
        self.argsNameModel = argsNameModel
        self.argsDebug = argsDebug
        self.argsTrainable = argsTrainable
        self.argsPreprocess = argsPreprocess
        self.printPrefix = printPrefix
    
    def setPathCSV(self, pathCSV):
        self.pathCSV = pathCSV
        
    def setDirBaseImg(self, dir_base_img):
        self.dirBaseImg = dir_base_img
        
    def __str__(self):
        return f"{self.__dict__}"