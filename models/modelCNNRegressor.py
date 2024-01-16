from entityModelConfig import ModelConfig
from models.modelABCRegressor import ModelABCRegressor
import tensorflow as tf

class ModelRegressorCNN(ModelABCRegressor):
    
    def __init__(self, modelConfig : ModelConfig):
        super().__init__(modelConfig)
        
    def getSpecialistModel(self):
        _model = tf.keras.models.Sequential([
                    # Camada de convolução 1
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                                           input_shape=(self.modelConfig.imageDimensionX, 
                                                        self.modelConfig.imageDimensionY, 
                                                        self.modelConfig.channelColors)),
                    
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(512, activation='relu'),
                    tf.keras.layers.Dense(1, activation='linear')
                    ])

        opt = tf.keras.optimizers.RMSprop()
        _model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mse'])

        _model.summary()
        return _model
    
    def reshapeTwoDimensions(self, X):
        return X
    
    def modelFit(self, model, X_, Y_carbono, X_validate, Y_carbono_validate):
        earlyStopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=self.modelConfig.argsPatience, 
                    restore_best_weights=True)
        
        model.fit(X_, Y_carbono, validation_split=0.3, 
                    epochs=self.modelConfig.argsEpochs, 
                            callbacks=[earlyStopping])

        #model.save(filepath=self.modelConfig.argsNameModel, save_format='tf', overwrite=True)
        #print(f"{self.modelConfig.printPrefix} Model Saved!!!")