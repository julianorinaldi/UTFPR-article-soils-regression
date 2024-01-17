from entityModelConfig import ModelConfig
from models.modelABCRegressor import ModelABCRegressor
from modelSet import ModelSet
import pandas as pd
import tensorflow as tf

class ModelRegressorTransferLearning(ModelABCRegressor):
    
    def __init__(self, modelConfig : ModelConfig):
        super().__init__(modelConfig)
        
    def getSpecialistModel(self, hp):
        pretrained_model = self._selectTransferLearningModel(self.modelConfig)

        # Adicionando camadas personalizadas no topo do modelo
        x = pretrained_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
         
        if self.modelConfig.argsGridSearch:
            hp_units = hp.Int('num_dense_units', min_value=64, max_value=256, step=64)
            hp_layers = hp.Int('num_dense_layers', min_value=1, max_value=2)
            hp_dropout = hp.Float('dropuot_rate', min_value=0.2, max_value=0.5, step=0.1)

            for _ in range(hp_layers):
                x = tf.keras.layers.Dense(units=hp_units, activation='relu')(x)
                x = tf.keras.layers.Dropout(hp_dropout)(x)
                
            predictions = tf.keras.layers.Dense(1, activation=hp.Choice('activation', values=['linear', 'sigmoid']))(x)
            
        else:
            x = tf.keras.layers.Dense(160, activation='relu')(x)
            #x = tf.keras.layers.Dense(64, activation='relu')(x)
            predictions = tf.keras.layers.Dense(1, activation='linear')(x)
            #predictions = tf.keras.layers.Dense(1)(x)
            
            
        _model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=predictions)
      
        if self.modelConfig.argsGridSearch:
            opt = tf.keras.optimizers.RMSprop(learning_rate=hp.Choice('learning_rate', values=[0.0001,0.001,0.01]),
                                              weight_decay=hp.Choice('weight_decay', values=[1e-6, 1e-5, 1e-4, 1e-3]))
        else:
            opt = tf.keras.optimizers.RMSprop()
        
        _model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mse'])

        if (self.modelConfig.argsShowModel):
            print(f'{self.modelConfig.printPrefix}')
            print(_model.summary())
            print(f'{self.modelConfig.printPrefix}')
        
        return _model
    
    def reshapeTwoDimensions(self, X):
        return X
    
    def modelFit(self, model, X_, Y_carbono, X_validate, Y_carbono_validate):
        earlyStopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=self.modelConfig.argsPatience, 
                    restore_best_weights=True)
        
        if (not self.modelConfig.argsSepared):
            # Padrão sem separação entre validação e treino      
            X_ = pd.concat([X_, X_validate], axis=0)
            X_ = X_.reset_index(drop=True)
            Y_carbono = pd.concat([Y_carbono, Y_carbono_validate], axis=0)
            Y_carbono = Y_carbono.reset_index(drop=True)
            model.fit(X_, Y_carbono, validation_split=0.3, epochs=self.modelConfig.argsEpochs, 
                            callbacks=[earlyStopping])
        else:
            model.fit(X_, Y_carbono, validation_data=(X_validate, Y_carbono_validate),
                    epochs=self.modelConfig.argsEpochs, 
                            callbacks=[earlyStopping])
        
        #model.save(filepath=self.modelConfig.argsNameModel, save_format='tf', overwrite=True)
        #print(f"{self.modelConfig.printPrefix} Model Saved!!!")
        
    def _selectTransferLearningModel(self, modelConfig : ModelConfig):
        # Modelos disponíveis para Transfer-Learning
        # https://keras.io/api/applications/
        
        # include_top=False => Excluindo as camadas finais (top layers) da rede, que geralmente são usadas para classificação. Vamos adicionar nossas próprias camadas finais
        # input_shape=(imageDimensionX, imageDimensionY, qtd_canal_color) => Nosso array de imagens tem as dimensões (256, 256, 3)
        # pooling => Modo de pooling opcional para extração de recursos quando include_top for False [none, avg (default), max], passado por parâmetro, mas o default é avg.
        # classes=1 => Apenas uma classe de saída, no caso de regressão precisamos de valor predito para o carbono.
        # weights='imagenet' => Carregamento do modelo inicial com pesos do ImageNet, no qual no treinamento será re-adaptado.
        if modelConfig.modelSet == ModelSet.ResNet50:
            pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                            input_shape=(modelConfig.imageDimensionX, modelConfig.imageDimensionY, 
                                                                        modelConfig.channelColors), classes=1, weights='imagenet')
        elif modelConfig.modelSet == ModelSet.ResNet101:
            pretrained_model = tf.keras.applications.ResNet101(include_top=False,
                                                            input_shape=(modelConfig.imageDimensionX, modelConfig.imageDimensionY, 
                                                                        modelConfig.channelColors), classes=1, weights='imagenet')
        elif modelConfig.modelSet == ModelSet.ResNet152:
            pretrained_model = tf.keras.applications.ResNet152(include_top=False,
                                                            input_shape=(modelConfig.imageDimensionX, modelConfig.imageDimensionY, 
                                                                        modelConfig.channelColors), classes=1, weights='imagenet')
        elif modelConfig.modelSet == ModelSet.ConvNeXtBase:
            pretrained_model = tf.keras.applications.ConvNeXtBase(include_top=False,
                                                            input_shape=(modelConfig.imageDimensionX, modelConfig.imageDimensionY, 
                                                                        modelConfig.channelColors), classes=1, weights='imagenet')
        elif modelConfig.modelSet == ModelSet.ConvNeXtXLarge:
            pretrained_model = tf.keras.applications.ConvNeXtXLarge(include_top=False,
                                                            input_shape=(modelConfig.imageDimensionX, modelConfig.imageDimensionY, 
                                                                        modelConfig.channelColors), classes=1, weights='imagenet')
        elif modelConfig.modelSet == ModelSet.EfficientNetB7:
            pretrained_model = tf.keras.applications.EfficientNetB7(include_top=False,
                                                            input_shape=(modelConfig.imageDimensionX, modelConfig.imageDimensionY, 
                                                                        modelConfig.channelColors), classes=1, weights='imagenet')
        elif modelConfig.modelSet == ModelSet.EfficientNetV2S:
            pretrained_model = tf.keras.applications.EfficientNetV2S(include_top=False,
                                                            input_shape=(modelConfig.imageDimensionX, modelConfig.imageDimensionY, 
                                                                        modelConfig.channelColors), classes=1, weights='imagenet')
        elif modelConfig.modelSet == ModelSet.EfficientNetV2L:
            pretrained_model = tf.keras.applications.EfficientNetV2L(include_top=False,
                                                            input_shape=(modelConfig.imageDimensionX, modelConfig.imageDimensionY, 
                                                                        modelConfig.channelColors), classes=1, weights='imagenet')
            
        elif modelConfig.modelSet == ModelSet.InceptionResNetV2:
            pretrained_model = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                            input_shape=(modelConfig.imageDimensionX, modelConfig.imageDimensionY, 
                                                                        modelConfig.channelColors), classes=1, weights='imagenet')
        elif modelConfig.modelSet == ModelSet.DenseNet169:
            pretrained_model = tf.keras.applications.DenseNet169(include_top=False,
                                                            input_shape=(modelConfig.imageDimensionX, modelConfig.imageDimensionY, 
                                                                        modelConfig.channelColors), classes=1, weights='imagenet')
        elif modelConfig.modelSet == ModelSet.VGG19:
            pretrained_model = tf.keras.applications.VGG19(include_top=False,
                                                            input_shape=(modelConfig.imageDimensionX, modelConfig.imageDimensionY, 
                                                                        modelConfig.channelColors), classes=1, weights='imagenet')
        else:
            raise Exception('Modelo desconhecido')


        pretrained_model.trainable = modelConfig.argsTrainable
        for layer in pretrained_model.layers:
            layer.trainable = modelConfig.argsTrainable
        
        # Faz com que o modelo seja treinado nos últimos 20 camadas
        # for layer in pretrained_model.layers[-1:-21:-1]:
        #     layer.trainable = True
        
        
        return pretrained_model
            
    
# Carregando Modelo
# resnet_model = tf.keras.models.load_model(filepath = self.modelConfig.argsNameModel)
# if (self.modelConfig.argsDebug):
#     print(f'{self.modelConfig.printPrefix}')
#     print(resnet_model.summary())
#     print(f'{self.modelConfig.printPrefix}')