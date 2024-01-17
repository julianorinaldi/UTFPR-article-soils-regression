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
    def modelFit(self, model, X_, Y_carbono, X_validate, Y_carbono_validate):
        # Juntando os dados de validação com treino no SUPER.
        X_ = pd.concat([X_, X_validate], axis=0)
        X_ = X_.reset_index(drop=True)
        Y_carbono = pd.concat([Y_carbono, Y_carbono_validate], axis=0)
        Y_carbono = Y_carbono.reset_index(drop=True)
        
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
        df_sorted.to_csv('resultado.csv', index=False)
        #print(f'{df_sorted.to_string(index=False)}')
        #print()
        print(f'{self.modelConfig.printPrefix} Melhores resultados ...')
        print(f'{df_sorted.head()}')
        print()
        print(f'{self.modelConfig.printPrefix} Piores resultados ...')
        print(f'{df_sorted.tail()}')
        print()
            
   
    def _load_images(self, modelConfig : ModelConfig, qtdImagens : int):
        df, imgFileNames, df_validate, imgFileNamesValidate = dataset_process(modelConfig)

        # Quantidade de imagens usadas para a rede.
        qtd_imagens = len(df)
        if (qtd_imagens > qtdImagens) and (qtdImagens > 0):
            qtd_imagens = qtdImagens

        if (modelConfig.argsDebug):
            print(f'{modelConfig.printPrefix} Dados de validação do Treino')
        # Array com as imagens a serem carregadas para validação do treino
        imageArrayValidate = image_load(modelConfig, imgFileNamesValidate, qtd_imagens)
        X_validate, Y_carbono_validate = image_convert_array(modelConfig, imageArrayValidate, df_validate, qtd_imagens)

        if (modelConfig.argsDebug):
            print(f'{modelConfig.printPrefix} Dados do Treino')
        # Array com as imagens a serem carregadas de treino
        imageArray = image_load(modelConfig, imgFileNames, qtd_imagens)
        X_, Y_carbono = image_convert_array(modelConfig, imageArray, df, qtd_imagens)
        
        # Retorno X_ e Y_carbono, DataFrame, e Lista de Imagens
        # X_validate, Y_carbono_validate, df_validate, imgFileNamesValidate relaciona do Validate do Treino
        return X_, Y_carbono, df, imgFileNames, X_validate, Y_carbono_validate, df_validate, imgFileNamesValidate
        
    def train(self):
        self.modelConfig.setDirBaseImg('dataset/images/treinamento-solo-256x256')
        self.modelConfig.setPathCSV('dataset/csv/Dataset256x256-Treino.csv')
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Carregando imagens para o treino/validação')
        X_, Y_carbono, df, imgFileNames, X_validate, Y_carbono_validate, df_validate, imgFileNamesValidate  = self._load_images(self.modelConfig, qtdImagens=self.modelConfig.amountImagesTrain)
        
        # Flatten das imagens
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Fazendo reshape')
        
        # Aceita apenas 2 dimensões.
        X_validate = self.reshapeTwoDimensions(X_validate)
        X_ = self.reshapeTwoDimensions(X_)
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Novo shape de X_validate: {X_validate.shape}')
            print(f'{self.modelConfig.printPrefix} Novo shape de X_: {X_.shape}')
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Criando modelo: {self.modelConfig.modelSet.name}')
        
        self.model = self.getSpecialistModel()

        # Treinar o modelo
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Iniciando o treino')
        self.modelFit(self.model, X_, Y_carbono, X_validate, Y_carbono_validate)
        
    def test(self):
        # Agora entra o Test
        self.modelConfig.setDirBaseImg('dataset/images/teste-solo-256x256')
        self.modelConfig.setPathCSV('dataset/csv/Dataset256x256-Teste.csv')
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Carregando imagens para o teste')
            
        X_, Y_carbono, df, imgFileNames, X_validate, Y_carbono_validate, df_validate, imgFileNamesValidate = self._load_images(self.modelConfig, qtdImagens=self.modelConfig.amountImagesTest)
        
        # No teste por ignorar estes dados, eles devem estar vazios.
        # X_validate, Y_carbono_validate, df_validate, imgFileNamesValidate
        
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

        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Alguns exemplos de predições ...')
            self._showPredictSamples(X_, imgFileNames, Y_carbono, prediction)
        
        print()
        print(f'====================================================')
        print(f'====================================================')
        print(f'=========>>>>> R2: {r2} <<<<<=========')
        print(f'====================================================')
        print(f'====================================================')
        print()
