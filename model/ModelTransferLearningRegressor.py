import numpy as np
import tensorflow as tf

from core.ModelConfig import ModelConfig
from core.ModelSetEnum import ModelSetEnum
from model.ModelABCRegressor import ModelABCRegressor

class ModelRegressorTransferLearning(ModelABCRegressor):
    
    def __init__(self, config : ModelConfig):
        super().__init__(config)
        
    def getSpecialistModel(self, hp):
        pretrained_model = self._selectTransferLearningModel()

        # Adicionando camadas personalizadas no topo do modelo
        x = pretrained_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Flatten()(x)
         
        if self.config.argsGridSearch:
            hp_dense1 = hp.Float('dense1', min_value=32, max_value=256, step=32)
            x = tf.keras.layers.Dense(hp_dense1, activation='relu')(x)
            hp_dropout1 = hp.Float('dropuot_rate1', min_value=0.3, max_value=0.5, step=0.1)
            x = tf.keras.layers.Dropout(hp_dropout1)(x)
                
            predictions = tf.keras.layers.Dense(1, activation=hp.Choice('activation', values=['linear']))(x)
            
        else:
            x = tf.keras.layers.Dense(160, activation='relu')(x)
            predictions = tf.keras.layers.Dense(1, activation='linear')(x)
            
        _model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=predictions)
      
        if self.config.argsGridSearch:
            opt = tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[0.0001]))
        else:
            opt = tf.keras.optimizers.RMSprop()
        
        _model.compile(optimizer=opt, loss='mae', metrics=['mae', 'mse'])

        if (self.config.argsShowModel):
            self.config.logger.logInfo(f"{_model.summary()}")

        return _model
    
    def reshapeTwoDimensions(self, X):
        return X
    
    def modelFit(self, models, X_, Y_carbono, X_validate, Y_carbono_validate):
        earlyStopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_mae', patience=self.config.argsPatience, 
                    restore_best_weights=True)
        
        for model in models:
            if (not self.config.argsSepared):
                # Padrão sem separação entre validação e treino      
                X_ = np.concatenate((X_, X_validate), axis=0)
                Y_carbono = np.concatenate((Y_carbono, Y_carbono_validate), axis=0)
                model.fit(X_, Y_carbono, validation_split=0.2, epochs=self.config.argsEpochs, 
                                callbacks=[earlyStopping])
            else:
                model.fit(X_, Y_carbono, validation_data=(X_validate, Y_carbono_validate),
                        epochs=self.config.argsEpochs, 
                                callbacks=[earlyStopping])
            
            model.save(filepath=self.config.argsNameModel, save_format='tf', overwrite=True)
            self.config.logger.logInfo(f"Model Saved!!!")
        
    def _selectTransferLearningModel(self):
        # Modelos disponíveis para Transfer-Learning
        # https://keras.io/api/applications/
        
        # include_top=False => Excluindo as camadas finais (top layers) da rede, que geralmente são usadas para classificação. Vamos adicionar nossas próprias camadas finais
        # input_shape=(imageDimensionX, imageDimensionY, qtd_canal_color) => Nosso array de imagens tem as dimensões (256, 256, 3)
        # pooling => Modo de pooling opcional para extração de recursos quando include_top for False [none, avg (default), max], passado por parâmetro, mas o default é avg.
        # classes=1 => Apenas uma classe de saída, no caso de regressão precisamos de valor predito para o carbono.
        # weights='imagenet' => Carregamento do modelo inicial com pesos do ImageNet, no qual no treinamento será re-adaptado.
        if self.config.modelSetEnum == ModelSetEnum.ResNet50:
            pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                            input_shape=(self.config.imageDimensionX, self.config.imageDimensionY, 
                                                                        self.config.channelColors), classes=1, weights='imagenet')
        elif self.config.modelSetEnum == ModelSetEnum.ResNet101:
            pretrained_model = tf.keras.applications.ResNet101(include_top=False,
                                                            input_shape=(self.config.imageDimensionX, self.config.imageDimensionY, 
                                                                        self.config.channelColors), classes=1, weights='imagenet')
        elif self.config.modelSetEnum == ModelSetEnum.ResNet152:
            pretrained_model = tf.keras.applications.ResNet152(include_top=False,
                                                            input_shape=(self.config.imageDimensionX, self.config.imageDimensionY, 
                                                                        self.config.channelColors), classes=1, weights='imagenet')
        elif self.config.modelSetEnum == ModelSetEnum.ConvNeXtBase:
            pretrained_model = tf.keras.applications.ConvNeXtBase(include_top=False,
                                                            input_shape=(self.config.imageDimensionX, self.config.imageDimensionY, 
                                                                        self.config.channelColors), classes=1, weights='imagenet')
        elif self.config.modelSetEnum == ModelSetEnum.ConvNeXtXLarge:
            pretrained_model = tf.keras.applications.ConvNeXtXLarge(include_top=False,
                                                            input_shape=(self.config.imageDimensionX, self.config.imageDimensionY, 
                                                                        self.config.channelColors), classes=1, weights='imagenet')
        elif self.config.modelSetEnum == ModelSetEnum.EfficientNetB7:
            pretrained_model = tf.keras.applications.EfficientNetB7(include_top=False,
                                                            input_shape=(self.config.imageDimensionX, self.config.imageDimensionY, 
                                                                        self.config.channelColors), classes=1, weights='imagenet')
        elif self.config.modelSetEnum == ModelSetEnum.EfficientNetV2S:
            pretrained_model = tf.keras.applications.EfficientNetV2S(include_top=False,
                                                            input_shape=(self.config.imageDimensionX, self.config.imageDimensionY, 
                                                                        self.config.channelColors), classes=1, weights='imagenet')
        elif self.config.modelSetEnum == ModelSetEnum.EfficientNetV2L:
            pretrained_model = tf.keras.applications.EfficientNetV2L(include_top=False,
                                                            input_shape=(self.config.imageDimensionX, self.config.imageDimensionY, 
                                                                        self.config.channelColors), classes=1, weights='imagenet')
            
        elif self.config.modelSetEnum == ModelSetEnum.InceptionResNetV2:
            pretrained_model = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                            input_shape=(self.config.imageDimensionX, self.config.imageDimensionY, 
                                                                        self.config.channelColors), classes=1, weights='imagenet')
        elif self.config.modelSetEnum == ModelSetEnum.DenseNet169:
            pretrained_model = tf.keras.applications.DenseNet169(include_top=False,
                                                            input_shape=(self.config.imageDimensionX, self.config.imageDimensionY, 
                                                                        self.config.channelColors), classes=1, weights='imagenet')
        elif self.config.modelSetEnum == ModelSetEnum.VGG19:
            pretrained_model = tf.keras.applications.VGG19(include_top=False,
                                                            input_shape=(self.config.imageDimensionX, self.config.imageDimensionY, 
                                                                        self.config.channelColors), classes=1, weights='imagenet')
        else:
            raise Exception('Modelo desconhecido')


        pretrained_model.trainable = self.config.argsTrainable
        for layer in pretrained_model.layers:
            layer.trainable = self.config.argsTrainable
        
        # Faz com que o modelo seja treinado nos últimos 20 camadas
        # for layer in pretrained_model.layers[-1:-21:-1]:
        #     layer.trainable = True
        
        
        return pretrained_model
            
    
# Carregando Modelo
# resnet_model = tf.keras.models.load_model(filepath = self.config.argsNameModel)
# if (self.config.argsDebug):
#     self.config.logger.logInfo(f"{resnet_model.summary()}")
