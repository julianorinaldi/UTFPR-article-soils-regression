import pandas as pd
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from abc import ABC, abstractmethod
from core.ModelConfig import ModelConfig
from core.DatasetProcess import DatasetProcess
from core.ImageProcess import ImageProcess
from sklearn.metrics import r2_score
from keras_tuner.tuners import RandomSearch
from sklearn.model_selection import GridSearchCV

class ModelABCRegressor(ABC):
    def __init__(self, config : ModelConfig):
        self.config = config
        self.models = []
        self.hyperparameters = []
        super().__init__()

    # Implemente para cada modelo de algoritmo de machine learn
    @abstractmethod
    def getSpecialistModel(self, hp):
        pass
    
    # Implemente se não desejar converter em 2 dimensões
    # Padrão que vem: (qtdImage, 256,256,3)
    # Na implementação abaixo, fica: (qtdImage, 196608) usado para algoritmos padrões
    @abstractmethod
    def reshapeTwoDimensions(self, X):
        return X.reshape(X.shape[0], -1)
    
    # Re-implemente se desejar fazer um fit diferente, por exempĺo para CNN
    @abstractmethod
    def modelFit(self, models, X_, Y_carbono, X_validate, Y_carbono_validate):
        # Juntando os dados de validação com treino no SUPER.
        X_ = np.concatenate((X_, X_validate), axis=0)
        Y_carbono = np.concatenate((Y_carbono, Y_carbono_validate), axis=0)
        
        for model in models:
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
        #df_sorted.to_csv('resultado.csv', index=False)
        #self.config.logger.logInfo(f"df_sorted.to_string(index=False)}")
        
        df_sorted['grupo'] = df_sorted['amostra'].str.extract(r'([A-Z]+\d+)')[0]
        
        self.config.logger.logInfo(f"")
        self.config.logger.logInfo(f"Melhores resultados ...")
        self.config.logger.logInfo(f"{df_sorted.head()}")
        self.config.logger.logInfo(f"")
        self.config.logger.logInfo(f"Piores resultados ...")
        self.config.logger.logInfo(f"{df_sorted.tail()}")
        self.config.logger.logInfo(f"")
        
        df_media = df_sorted.groupby('grupo').agg({'teor_cabono_predict': 'mean', 'teor_cabono_real': 'first'}).reset_index()
        r2 = r2_score(df_media['teor_cabono_real'], df_media['teor_cabono_predict'])
        
        self.config.logger.logInfo(f"R2 sobre a média de predição:")
        self.config.logger.logInfo(f"")
        self.config.logger.logInfo(f"====================================================")
        self.config.logger.logInfo(f"====================================================")
        self.config.logger.logInfo(f"=========>>>>> R2 da média: {r2} <<<<<=========")
        self.config.logger.logInfo(f"====================================================")
        self.config.logger.logInfo(f"====================================================")
        self.config.logger.logInfo(f"")
   
    def _load_images(self, qtdImagens : int):
        datasetProcess = DatasetProcess(self.config)
        df, imgFileNames, df_validate, imgFileNamesValidate = datasetProcess.dataset_process()

        # Quantidade de imagens usadas para a rede.
        qtd_imagens = len(df)
        if (qtd_imagens > qtdImagens) and (qtdImagens > 0):
            qtd_imagens = qtdImagens

        self.config.logger.logInfo(f"Dados de validação do Treino")
        imageProcess = ImageProcess(self.config)
        # Array com as imagens a serem carregadas para validação do treino
        imageArrayValidate = imageProcess.image_load(imgFileNamesValidate, qtd_imagens)
        X_validate, Y_carbono_validate = imageProcess.image_convert_array(imageArrayValidate, df_validate, qtd_imagens)

        self.config.logger.logInfo(f"Dados do Treino")
        # Array com as imagens a serem carregadas de treino
        imageArray = imageProcess.image_load(imgFileNames, qtd_imagens)
        X_, Y_carbono = imageProcess.image_convert_array(imageArray, df, qtd_imagens)
        
        # Retorno X_ e Y_carbono, DataFrame, e Lista de Imagens
        # X_validate, Y_carbono_validate, df_validate, imgFileNamesValidate relaciona do Validate do Treino
        return X_, Y_carbono, X_validate, Y_carbono_validate, imgFileNames
        
    def train(self):
        self.config.setDirBaseImg('dataset/images/treinamento-solo-256x256')
        self.config.setPathCSV('dataset/csv/Dataset256x256-Treino.csv')
        
        self.config.logger.logInfo(f"Carregando imagens para o treino/validação")
        X_, Y_carbono, X_validate, Y_carbono_validate, imgFileNames = self._load_images(qtdImagens=self.config.amountImagesTrain)
        
        # Flatten das imagens
        self.config.logger.logDebug(f"Fazendo reshape")
        
        # Aceita apenas 2 dimensões.
        X_validate = self.reshapeTwoDimensions(X_validate)
        X_ = self.reshapeTwoDimensions(X_)
        
        self.config.logger.logDebug(f"Novo shape de X_validate: {X_validate.shape}")
        self.config.logger.logDebug(f"Novo shape de X_: {X_.shape}")
        
        self.config.logger.logInfo(f"Criando modelo: {self.config.modelSetEnum.name}")
        
        # Treinar o modelo
        self.config.logger.logInfo(f"Iniciando o treino")
            
        if (not self.config.argsGridSearch > 0):
            # Executa sem GridSearch
            self.config.logger.logInfo(f"Executando sem o GridSearch")
            self.models = { self.getSpecialistModel(hp = None) }
            self.modelFit(self.models, X_, Y_carbono, X_validate, Y_carbono_validate)
        else:
            # Executa com GridSearch
            self.config.logger.logInfo(f"Executando com o GridSearch")
            earlyStopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                            patience=self.config.argsPatience, restore_best_weights=True)
            tuner = RandomSearch(
                self.getSpecialistModel,
                objective='val_loss',
                max_trials=self.config.argsGridSearch,  # Quantas tentativas de hiperparâmetros serão executadas
                directory='_GridSearchTuning',  # diretório para armazenar os resultados
                project_name='RandomSearchTuning'
            )
            
            if (not self.config.argsSepared):
                # Padrão sem separação entre validação e treino      
                X_ = np.concatenate((X_, X_validate), axis=0)
                Y_carbono = np.concatenate((Y_carbono, Y_carbono_validate), axis=0)
                tuner.search(X_, Y_carbono, epochs=self.config.argsEpochs, 
                            validation_split=0.2, callbacks=[earlyStopping])
            else:
                # Execute a busca de hiperparâmetros
                tuner.search(X_, Y_carbono, epochs=self.config.argsEpochs, 
                            validation_data=(X_validate, Y_carbono_validate),
                            callbacks=[earlyStopping])

            self.hyperparameters = tuner.get_best_hyperparameters(num_trials=self.config.argsGridSearch)
            # Imprima os melhores hiperparâmetros encontrados
            self.config.logger.logInfo(f"Melhores Hyperparameters:")
            self.config.logger.logInfo(f"{self.hyperparameters[0].values}")
            
            # Obtenha a melhor tentativa
            best_trial = tuner.oracle.get_best_trials(num_trials=self.config.argsGridSearch)
            _models = []
            for trial in best_trial:
                _models.append(tuner.load_model(trial))    
            self.models = _models
            del tuner
        
    def test(self):
        # Agora entra o Test
        self.config.setDirBaseImg('dataset/images/teste-solo-256x256')
        self.config.setPathCSV('dataset/csv/Dataset256x256-Teste.csv')
        
        self.config.logger.logInfo(f"Carregando imagens para o teste")
            
        X_, Y_carbono, X_validate, Y_carbono_validate, imgFileNames = self._load_images(qtdImagens=self.config.amountImagesTest)
        
        # No teste por ignorar estes dados, eles devem estar vazios.
        # X_validate, Y_carbono_validate, df_validate, imgFileNamesValidate
        
        # Aceita apenas 2 dimensões.
        X_ = self.reshapeTwoDimensions(X_)
        
        self.config.logger.logInfo(f"Iniciando predição completa para o R2...")
        
        for index, model in enumerate(self.models):
            # Fazendo a predição sobre os dados de teste
            prediction = model.predict(X_) # type: ignore

            # Avaliando com R2
            r2 = r2_score(Y_carbono, prediction)

            self.config.logger.logInfo(f"")
            self.config.logger.logInfo(f"====================================================")
            self.config.logger.logInfo(f"====================================================")
            self.config.logger.logInfo(f"=========>>>>> R2 Modelo: {r2} <<<<<=========")
            self.config.logger.logInfo(f"====================================================")
            self.config.logger.logInfo(f"====================================================")
            self.config.logger.logInfo(f"")

            self.config.logger.logInfo(f"Hiperparâmetros deste modelo:")
            self.config.logger.logInfo(f"{self.hyperparameters[index].values}")
            self.config.logger.logInfo(f"")
            
            self.config.logger.logInfo(f"Alguns exemplos de predições ...")
            self._showPredictSamples(X_, imgFileNames, Y_carbono, prediction)
                
            del model
        del self.models
            

