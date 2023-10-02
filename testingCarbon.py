import numpy as np  # Trabalhar com array
import tensorflow as tf  # Trabalhar com aprendizado de máquinas
import random

from sklearn.metrics import r2_score  # Avaliação das Métricas
from imageProcess import image_load, image_convert_array, image_processing
from datasetProcess import dataset_process
from entityModelConfig import ModelConfig

class TestCarbon:
    def __init__(self, modelConfig : ModelConfig):
        self.modelConfig = modelConfig
        
    def test(self):
        df, imageNamesList = dataset_process(self.modelConfig)

        # Quantidade de imagens usadas para a rede.
        qtd_imagens = len(df)
        if (qtd_imagens > self.modelConfig.amountImagesTest):
            qtd_imagens = self.modelConfig.amountImagesTest
            
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Preprocess: {self.modelConfig.argsPreprocess}')
        
        # Array com as imagens a serem carregadas de treino
        imageArray = image_load(self.modelConfig, imageNamesList, qtd_imagens)

        X_, Y_carbono = image_convert_array(self.modelConfig, imageArray, df, qtd_imagens)

        # Carregando Modelo
        resnet_model = tf.keras.models.load_model(filepath = self.modelConfig.argsNameModel)
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix}')
            print(resnet_model.summary())
            print(f'{self.modelConfig.printPrefix}')

        # Trazendo algumas amostras aleatórias ...
        for i in [1, 5, 10, 50, 60, 100, 200, 300, 400, 500, 1000, 2000, 3000, 3500]:
            # Essa linha abaixo garante aleatoriedade
            indexImg = random.randint(0, i)
            img_path = f'{imageNamesList[indexImg]}'
            img = image_processing(self.modelConfig, img_path)
            img = np.expand_dims(img, axis=0)

            predictValue = resnet_model.predict(img)
            Real = df.teor_carbono[indexImg]

            print(f'{self.modelConfig.printPrefix} Original image[{indexImg}]: {imageNamesList[indexImg]} => {df.teor_carbono[indexImg]}')
            print(f'{self.modelConfig.printPrefix} {self.modelConfig.modelSet.name}[{indexImg}]: {predictValue.item(0)} => Diff: {Real - predictValue.item(0)}')
            print("")

        # Fazendo a predição sobre os dados de teste
        prediction = resnet_model.predict(X_)

        # Avaliando com R2
        r2 = r2_score(Y_carbono, prediction)
        print()
        print(f'====================================================')
        print(f'====================================================')
        print(f'=========>>>>> R2: {r2} <<<<<=========')
        print(f'====================================================')
        print(f'====================================================')
