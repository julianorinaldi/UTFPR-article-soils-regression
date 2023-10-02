from entityModelConfig import ModelConfig
from models.modelABCRegressor import ModelABCRegressor
from modelSet import ModelSet
import tensorflow as tf

class ModelRegressorTransferLearning(ModelABCRegressor):
    
    def __init__(self, modelConfig : ModelConfig):
        super().__init__(modelConfig)
        
    def getSpecialistModel(self):
        pretrained_model = self._selectTransferLearningModel(self.modelConfig)

        # Adicionando camadas personalizadas no topo do modelo
        x = pretrained_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        #x = tf.keras.layers.Flatten()(x)
        #x = tf.keras.layers.Dense(512, activation='relu')(x)
        predictions = tf.keras.layers.Dense(1, activation='linear')(x)
        _model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=predictions)
      
        opt = tf.keras.optimizers.RMSprop()
        _model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mse'])

        print(f'{self.modelConfig.printPrefix}')
        print(_model.summary())
        print(f'{self.modelConfig.printPrefix}')
        
        return _model
    
    def reshapeTwoDimensions(self, X):
        return X
    
    def modelFit(self, model, X_, Y_carbono):
        earlyStopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=self.modelConfig.argsPatience, 
                    restore_best_weights=True)
        
        model.fit(X_, Y_carbono, validation_split=0.3, 
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
        elif modelConfig.modelSet == ModelSet.EfficientNetB7:
            pretrained_model = tf.keras.applications.EfficientNetB7(include_top=False,
                                                            input_shape=(modelConfig.imageDimensionX, modelConfig.imageDimensionY, 
                                                                        modelConfig.channelColors), classes=1, weights='imagenet')
        elif modelConfig.modelSet == ModelSet.EfficientNetV2S:
            pretrained_model = tf.keras.applications.EfficientNetV2S(include_top=False,
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
        
        return pretrained_model
            
    
# Carregando Modelo
# resnet_model = tf.keras.models.load_model(filepath = self.modelConfig.argsNameModel)
# if (self.modelConfig.argsDebug):
#     print(f'{self.modelConfig.printPrefix}')
#     print(resnet_model.summary())
#     print(f'{self.modelConfig.printPrefix}')