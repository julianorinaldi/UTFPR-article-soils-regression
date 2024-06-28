from core.ModelConfig import ModelConfig
from model.ModelABCRegressor import ModelABCRegressor
import pandas as pd
import tensorflow as tf


class ModelRegressorCNN(ModelABCRegressor):

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def get_specialist_model(self, hp):
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

        if self.config.argsShowModel:
            self.config.logger.log_info(f"{_model.summary()}")

        return _model

    def reshape_two_dimensions(self, x_data):
        return x_data

    def model_fit(self, models, x_data, y_carbono, x_validate, y_carbono_validate):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.config.argsPatience,
            restore_best_weights=True)

        for model in models:
            if not self.config.argsSepared:
                # Padrão sem separação entre validação e treino      
                x_data = pd.concat([x_data, x_validate], axis=0)
                x_data = x_data.reset_index(drop=True)
                y_carbono = pd.concat([y_carbono, y_carbono_validate], axis=0)
                y_carbono = y_carbono.reset_index(drop=True)
                model.fit(x_data, y_carbono, validation_split=0.3, epochs=self.config.argsEpochs,
                          callbacks=[early_stopping])
            else:
                model.fit(x_data, y_carbono, validation_data=(x_validate, y_carbono_validate),
                          epochs=self.config.argsEpochs, callbacks=[early_stopping])

            #model.save(filepath=self.modelConfig.argsNameModel, save_format='tf', overwrite=True)
            #self.config.logger.logInfo(f"Model Saved!!!")
