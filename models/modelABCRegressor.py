import numpy as np  # Trabalhar com array
import random

from abc import ABC, abstractmethod
from entityModelConfig import ModelConfig
from datasetProcess import dataset_process
from imageProcess import image_load, image_convert_array
from sklearn.metrics import r2_score
from imageProcess import image_processing

class ModelABCRegressor(ABC):
    def __init__(self, modelConfig : ModelConfig):
        self.modelConfig = modelConfig
        self.model = None
        super().__init__()

    # Implemente para cada modelo de algoritmo de machine learn
    @abstractmethod
    def getSpecialistModel(self):
        pass
    
    # Implemente se não desejar converter em 2 dimensões
    # Padrão que vem: (qtdImage, 256,256,3)
    # Na implementação abaixo, fica: (qtdImage, 196608) usado para algoritmos padrões
    @abstractmethod
    def reshapeTwoDimensions(self, X):
        return X.reshape(X.shape[0], -1)
    
    # Re-implemente se desejar fazer um fit diferente, por exempĺo para CNN
    @abstractmethod
    def modelFit(self, model, X_, Y_carbono):
        model.fit(X_, Y_carbono)
    
    def modelPredictTest(self, model, df, imageNamesList):
        # Trazendo algumas amostras aleatórias ...
        for i in [1, 100, 500, 1000, 2000, 3000]:
            # Essa linha abaixo garante aleatoriedade
            indexImg = random.randint(0, i)
            if (indexImg >= len(imageNamesList)):
                indexImg = random.randint(0, len(imageNamesList) - 1)
            img_path = f'{imageNamesList[indexImg]}'
            img = image_processing(self.modelConfig, img_path)
            img = np.expand_dims(img, axis=0)
            img = self.reshapeTwoDimensions(img)
            predictValue = model.predict(img)
            Real = df.teor_carbono[indexImg]

            print(f'{self.modelConfig.printPrefix} Original image[{indexImg}]: {imageNamesList[indexImg]} => {df.teor_carbono[indexImg]}')
            print(f'{self.modelConfig.printPrefix} {self.modelConfig.modelSet.name}[{indexImg}]: {predictValue.item(0)} => Diff: {Real - predictValue.item(0)}')
            print("")
    
    
    def _load_images(self, modelConfig : ModelConfig, qtdImagens : int):
        df, imageNamesList = dataset_process(modelConfig)

        # Quantidade de imagens usadas para a rede.
        qtd_imagens = len(df)
        if (qtd_imagens > qtdImagens) and (qtdImagens > 0):
            qtd_imagens = qtdImagens

        # Array com as imagens a serem carregadas de treino
        imageArray = image_load(modelConfig, imageNamesList, qtd_imagens)

        X_, Y_carbono = image_convert_array(modelConfig, imageArray, df, qtd_imagens)
        
        # Retorno X_ e Y_carbono, DataFrame, e Lista de Imagens
        return X_, Y_carbono, df, imageNamesList
        
    def train(self):
        self.modelConfig.setDirBaseImg('dataset/images/treinamento-solo-256x256')
        self.modelConfig.setPathCSV('dataset/csv/Dataset256x256-Treino.csv')
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Carregando imagens para o treino')
        X_, Y_carbono, df, imageNamesList  = self._load_images(self.modelConfig, qtdImagens=self.modelConfig.amountImagesTrain)
        
        # Flatten das imagens
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Fazendo reshape')
        
        # Aceita apenas 2 dimensões.
        X_ = self.reshapeTwoDimensions(X_)
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Novo shape de X_: {X_.shape}')
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Criando modelo: {self.modelConfig.modelSet.name}')
        
        self.model = self.getSpecialistModel()

        # Treinar o modelo
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Iniciando o treino')
        self.modelFit(self.model, X_, Y_carbono)
        
    def test(self):
        # Agora entra o Test
        self.modelConfig.setDirBaseImg('dataset/images/teste-solo-256x256')
        self.modelConfig.setPathCSV('dataset/csv/Dataset256x256-Teste.csv')
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Carregando imagens para o teste')
            
        X_, Y_carbono, df, imageNamesList = self._load_images(self.modelConfig, qtdImagens=self.modelConfig.amountImagesTest)
        
        # Aceita apenas 2 dimensões.
        X_ = self.reshapeTwoDimensions(X_)
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Novo shape de X_: {X_.shape}')
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Iniciando predição completa para o R2...')
        
        # Fazendo a predição sobre os dados de teste
        prediction = self.model.predict(X_, batch_size=50)

        # Avaliando com R2
        r2 = r2_score(Y_carbono, prediction)
        print()
        print(f'====================================================')
        print(f'====================================================')
        print(f'=========>>>>> R2: {r2} <<<<<=========')
        print(f'====================================================')
        print(f'====================================================')
        print()
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Alguns exemplos de predições...')
        self.modelPredictTest(self.model, df, imageNamesList)
        