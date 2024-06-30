import os
from datetime import timezone, timedelta, datetime

from dto.ConfigModelDTO import ConfigModelDTO
from dto.FitDTO import FitDTO
from model.abstract.ModelABCRegressor import ModelABCRegressor
import pandas as pd
import tensorflow as tf


class ModelRegressorCNN(ModelABCRegressor):

    def __init__(self, config: ConfigModelDTO):
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

    def model_fit(self, models, fit_dto: FitDTO):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.config.argsPatience,
            restore_best_weights=True)

        for model in models:
            if not self.config.argsSepared:
                # Padrão sem separação entre validação e treino      
                x_img_data = pd.concat([fit_dto.x_img_train, fit_dto.x_img_validate], axis=0)
                x_img_data = x_img_data.reset_index(drop=True)
                y_df_data = pd.concat([fit_dto.y_df_train, fit_dto.y_df_validate], axis=0)
                y_df_data = y_df_data.reset_index(drop=True)
                model.fit(x_img_data, y_df_data, validation_split=0.3, epochs=self.config.argsEpochs,
                          callbacks=[early_stopping])
            else:
                model.fit(fit_dto.x_img_train, fit_dto.y_df_train, validation_data=(fit_dto.x_img_validate, fit_dto.y_df_validate),
                          epochs=self.config.argsEpochs, callbacks=[early_stopping])

            tz_utc_minus3 = timezone(timedelta(hours=-3))
            timestamp = datetime.now(tz=tz_utc_minus3).strftime("%Y%m%d_%H%M")
            os.makedirs('out/model', exist_ok=True)
            filepath_model = f'out/model/CNN_{self.config.argsNameModel}_{timestamp}.keras'
            model.save(filepath=filepath_model, overwrite=True)
            self.config.logger.log_info(f"Model Saved!!!")
