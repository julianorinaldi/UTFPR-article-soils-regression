import pandas as pd
import numpy as np  # Trabalhar com array
import random
from tqdm import tqdm

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
    
    def _showPredictSamples(self, carbonoImageArray, imgFileNames, cabonoRealArray, carbonoPredictionArray):
        self._minMaxPredictTest(carbonoImageArray, imgFileNames, cabonoRealArray, carbonoPredictionArray)

    def _minMaxPredictTest(self, carbonoImageArray, imgFileNames, cabonoRealArray, carbonoPredictionArray):
        result = []
        for i in tqdm(range(len(cabonoRealArray))):
            predictValue = carbonoPredictionArray[i]
            real = cabonoRealArray[i]
            diff = abs(real - predictValue)
            amostra = imgFileNames[i]
            erro = abs(diff)/abs(real)*100

            regLine = {'amostra': amostra, 'teor_cabono_real': real, 'teor_cabono_predict': predictValue, 'teor_cabono_diff' : diff, 'error(%)' : erro}
            result.append(regLine)
            
        df_sorted = pd.DataFrame(result)
        df_sorted = df_sorted.sort_values(by='error(%)')
        print(f'{df_sorted.to_string(index=False)}')
        print()
        print(f'{self.modelConfig.printPrefix} Melhores resultados ...')
        print(f'{df_sorted.head()}')
        print()
        print(f'{self.modelConfig.printPrefix} Piores resultados ...')
        print(f'{df_sorted.tail()}')
        print()
            
   
    def _load_images(self, modelConfig : ModelConfig, qtdImagens : int):
        df, imgFileNames = dataset_process(modelConfig)

        # Quantidade de imagens usadas para a rede.
        qtd_imagens = len(df)
        if (qtd_imagens > qtdImagens) and (qtdImagens > 0):
            qtd_imagens = qtdImagens

        # Array com as imagens a serem carregadas de treino
        imageArray = image_load(modelConfig, imgFileNames, qtd_imagens)

        X_, Y_carbono = image_convert_array(modelConfig, imageArray, df, qtd_imagens)
        
        # Retorno X_ e Y_carbono, DataFrame, e Lista de Imagens
        return X_, Y_carbono, df, imgFileNames
        
    def train(self):
        self.modelConfig.setDirBaseImg('dataset/images/treinamento-solo-256x256')
        self.modelConfig.setPathCSV('dataset/csv/Dataset256x256-Treino.csv')
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Carregando imagens para o treino')
        X_, Y_carbono, df, imgFileNames  = self._load_images(self.modelConfig, qtdImagens=self.modelConfig.amountImagesTrain)
        
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
        # Pensado em 06/10/2023 - Em estudo a DSA
        # 1) Talvez separar os dados de validação seja uma oportunidade de melhora no algoritmo.
        # 2) Outra estratégia é usar DataArgumentation para melhorar quantidade de imagens.
        # 3) Método de Ensamble - Talvez fazendo a predição do resultado de teste, seria interessante verificar a média
        # do % de carbono, entre o resultado de teste de todas a amostras em comum (de um mesmo grupo, da imagem original).
        # 4) Entender melhor sobre regularização L1, regularização Lasso, dropout
        # Pensado em 24/10/2023 - Voltar a estudo de attributes selection
        # 5) Analisar se é possível aplicar uma estratégia de seleção de atributos
        
        # Agora entra o Test
        #self.modelConfig.setDirBaseImg('dataset/images/teste-solo-256x256')
        #self.modelConfig.setPathCSV('dataset/csv/Dataset256x256-Teste.csv')
        
        # Força fazer o teste com o dataset do próprio treino
        # Objetiva verificar quais amostras estão ruim.
        self.modelConfig.setDirBaseImg('dataset/images/treinamento-solo-256x256')
        self.modelConfig.setPathCSV('dataset/csv/Dataset256x256-Treino.csv')
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Carregando imagens para o teste')
            
        X_, Y_carbono, df, imgFileNames = self._load_images(self.modelConfig, qtdImagens=self.modelConfig.amountImagesTest)
        
        # Aceita apenas 2 dimensões.
        X_ = self.reshapeTwoDimensions(X_)
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Novo shape de X_: {X_.shape}')
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Iniciando predição completa para o R2...')
        
        # Fazendo a predição sobre os dados de teste
        prediction = self.model.predict(X_) # type: ignore
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Shape de prediction: {prediction.shape}')

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
            print(f'{self.modelConfig.printPrefix} Alguns exemplos de predições ...')
        self._showPredictSamples(X_, imgFileNames, Y_carbono, prediction)
