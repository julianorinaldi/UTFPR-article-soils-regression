from core.ModelConfig import ModelConfig
from model.ModelABCRegressor import ModelABCRegressor
import pandas as pd
import tensorflow as tf

class ModelRegressorCNN(ModelABCRegressor):
    
    def __init__(self, config : ModelConfig):
        super().__init__(config)
        
    def getSpecialistModel(self, hp):
        _model = tf.keras.models.Sequential([
                    # Camada de convolução 1
                    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', 
                                           input_shape=(self.config.imageDimensionX, 
                                                        self.config.imageDimensionY, 
                                                        self.config.channelColors)),
                    
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(512, activation='relu'),
                    tf.keras.layers.Dense(1, activation='linear')
                    ])

        opt = tf.keras.optimizers.RMSprop()
        _model.compile(optimizer=opt, loss='mse', metrics=['mae', 'mse'])

        if (self.config.argsShowModel):
            self.config.logger.logInfo(f"{_model.summary()}")
            
        return _model
    
    def reshapeTwoDimensions(self, X):
        return X
    
    def modelFit(self, models, X_, Y_carbono, X_validate, Y_carbono_validate):
        earlyStopping = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=self.config.argsPatience, 
                    restore_best_weights=True)

        for model in models:
            if (not self.config.argsSepared):
                # Padrão sem separação entre validação e treino      
                X_ = pd.concat([X_, X_validate], axis=0)
                X_ = X_.reset_index(drop=True)
                Y_carbono = pd.concat([Y_carbono, Y_carbono_validate], axis=0)
                Y_carbono = Y_carbono.reset_index(drop=True)
                model.fit(X_, Y_carbono, validation_split=0.3, epochs=self.config.argsEpochs, 
                                callbacks=[earlyStopping])
            else:
                model.fit(X_, Y_carbono, validation_data=(X_validate, Y_carbono_validate),
                        epochs=self.config.argsEpochs, callbacks=[earlyStopping])
                
            #model.save(filepath=self.modelConfig.argsNameModel, save_format='tf', overwrite=True)
            #self.config.logger.logInfo(f"Model Saved!!!")