# -*- coding: utf-8 -*-

import numpy as np  # Trabalhar com array
import tensorflow as tf  # Trabalhar com aprendizado de máquinas

from imageProcess import image_load, image_convert_array
from datasetProcess import dataset_process
from modelSet import ModelSet
from entityModelConfig import ModelConfig
from modelTransferLearningProcess import modelTransferLearningProcess

class TrainingCarbon:
    def __init__(self, modelConfig : ModelConfig):
        self.modelConfig = modelConfig
    
    def train(self):
        df, imageNamesList = dataset_process(self.modelConfig)

        # Quantidade de imagens usadas para a rede.
        qtd_imagens = len(df)
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Preprocess: {self.modelConfig.argsPreprocess}')

        # Array com as imagens a serem carregadas de treino
        imageArray = image_load(self.modelConfig, imageNamesList, qtd_imagens)

        X_, Y_carbono = image_convert_array(self.modelConfig, imageArray, df, qtd_imagens)

        # Faz a chamada da criação do modelo de Transferência
        pretrained_model = modelTransferLearningProcess(self.modelConfig)

        # Adicionando camadas personalizadas no topo do modelo
        x = pretrained_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512, activation='relu')(x)
        predictions = tf.keras.layers.Dense(1, activation='linear')(x)

        # Define o novo modelo combinando a ResNet50 com as camadas personalizadas
        model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=predictions)
            
        print(f'{self.modelConfig.printPrefix}')
        print(model.summary())
        print(f'{self.modelConfig.printPrefix}')

        # Otimizadores
        # https://keras.io/api/optimizers/
        # Usuais
        #  tf.keras.optimizers.Adam(learning_rate=0.0001)
        #  tf.keras.optimizers.RMSprop(learning_rate=0.0001)
        #  tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.9)
        #  tf.keras.optimizers.Nadam(learning_rate=0.0001)
        opt = tf.keras.optimizers.RMSprop(learning_rate=0.0001)

        model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mse'])
        history = model.fit(X_, Y_carbono, validation_split=0.3, epochs=1, callbacks=[
                                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)])

        model.save(filepath=self.modelConfig.argsNameModel, save_format='tf', overwrite=True)

        print(f"{self.modelConfig.printPrefix} Model Saved!!!")
        print()
