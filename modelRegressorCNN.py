
from entityModelConfig import ModelConfig
from datasetProcess import dataset_process
from imageProcess import image_load, image_convert_array
from sklearn.metrics import r2_score  # Avaliação das Métricas
from sklearn.model_selection import train_test_split
import tensorflow as tf

class ModelRegressorCNN:
    def __init__(self, modelConfig : ModelConfig):
        self.modelConfig = modelConfig
        self.model = None

    def _load_images(self, modelConfig : ModelConfig):
        df, imageNamesList = dataset_process(modelConfig)

        # Quantidade de imagens usadas para a rede.
        qtd_imagens = len(df)
        if (modelConfig.argsDebug):
            print(f'{modelConfig.printPrefix} Preprocess: {modelConfig.argsPreprocess}')

        # Array com as imagens a serem carregadas de treino
        imageArray = image_load(modelConfig, imageNamesList, qtd_imagens)

        X_, Y_carbono = image_convert_array(modelConfig, imageArray, df, qtd_imagens)
        
        return X_, Y_carbono
        
    def train(self):
        self.modelConfig.setDirBaseImg('dataset/images/treinamento-solo-256x256')
        self.modelConfig.setPathCSV('dataset/csv/Dataset256x256-Treino.csv')
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Carregando imagens para o treino')
        X_, Y_carbono = self._load_images(self.modelConfig)
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Criando modelo: {self.modelConfig.modelSet.name}')
        
        self.model = tf.keras.models.Sequential([
                    # Camada de convolução 1
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                                           input_shape=(self.modelConfig.imageDimensionX, 
                                                        self.modelConfig.imageDimensionY, 
                                                        self.modelConfig.channelColors)),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    
                    # Camada de convolução 2
                    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                    
                    # Camada de convolução 3
                    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),

                    # Camada de convolução 3
                    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),

                    # Camada de convolução 3
                    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
                    tf.keras.layers.MaxPooling2D((2, 2)),
                                                            
                    # Transformando os dados em um vetor
                    tf.keras.layers.Flatten(),
                    
                    # Camada totalmente conectada 1
                    tf.keras.layers.Dense(1024, activation='linear'),
                    
                    # Camada de saída
                    tf.keras.layers.Dense(1, activation='linear')  # Esta camada possui 1 neurônio para a regressão
                ])

        opt = tf.keras.optimizers.RMSprop()

        self.model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mse'])

        self.model.summary()
        
        # Treinar o modelo
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Iniciando o treino')
        self.model.fit(X_, Y_carbono, validation_split=0.3, epochs=self.modelConfig.argsEpochs, 
                            callbacks=[tf.keras.callbacks.EarlyStopping
                                    (monitor='val_loss', patience=self.modelConfig.argsPatience, restore_best_weights=True)])
        
        self.model.save(filepath=self.modelConfig.argsNameModel, save_format='tf', overwrite=True)

        print(f"{self.modelConfig.printPrefix} Model Saved!!!")
        print()

        
    def test(self):
        # Agora entra o Test
        self.modelConfig.setDirBaseImg('dataset/images/teste-solo-256x256')
        self.modelConfig.setPathCSV('dataset/csv/Dataset256x256-Teste.csv')
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Carregando imagens para o teste')
        X_, Y_carbono = self._load_images(self.modelConfig)
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Iniciando predição...')
        # Fazendo a predição sobre os dados de teste
        prediction = self.model.predict(X_)

        # Avaliando com R2
        r2 = r2_score(Y_carbono, prediction)
        print()
        print(f'====================================================')
        print(f'====================================================')
        print(f'=========>>>>> R2: {r2} <<<<<=========')
        print(f'====================================================')
        print(f'====================================================')