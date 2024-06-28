import numpy as np
import tensorflow as tf

from core.ModelConfig import ModelConfig
from core.ModelSetEnum import ModelSetEnum
from model.ModelABCRegressor import ModelABCRegressor


class ModelRegressorTransferLearning(ModelABCRegressor):

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def get_specialist_model(self, hp):
        pretrained_model = self._select_transfer_learning_model

        # Adicionando camadas personalizadas no topo do modelo
        x = pretrained_model.output
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        if self.config.argsGridSearch:
            hp_dense1 = hp.Float('dense1', min_value=32, max_value=256, step=32)
            x = tf.keras.layers.Dense(hp_dense1, activation='relu')(x)
            hp_dropout1 = hp.Float('dropuot_rate1', min_value=0.3, max_value=0.5, step=0.1)
            x = tf.keras.layers.Dropout(hp_dropout1)(x)

            predictions = tf.keras.layers.Dense(2, activation=hp.Choice('activation', values=['linear']))(x)
        else:
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            predictions = tf.keras.layers.Dense(2, activation='linear')(x)

        _model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=predictions)

        if self.config.argsGridSearch:
            opt = tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[0.0001]))
        else:
            opt = tf.keras.optimizers.RMSprop()

        _model.compile(optimizer=opt, loss='mse', metrics=['mae'])

        if self.config.argsShowModel:
            self.config.logger.log_info(f"{_model.summary()}")

        return _model

    def reshape_two_dimensions(self, x_data):
        return x_data

    def model_fit(self, models, x_data, y_carbono, x_validate, y_carbono_validate):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_mae', patience=self.config.argsPatience,
            restore_best_weights=True)

        for model in models:
            if not self.config.argsSepared:
                # Padrão sem separação entre validação e treino      
                x_data = np.concatenate((x_data, x_validate), axis=0)
                y_carbono = np.concatenate((y_carbono, y_carbono_validate), axis=0)
                model.fit(x_data, y_carbono, validation_split=0.2, epochs=self.config.argsEpochs,
                          callbacks=[early_stopping])
            else:
                model.fit(x_data, y_carbono, validation_data=(x_validate, y_carbono_validate),
                          epochs=self.config.argsEpochs,
                          callbacks=[early_stopping])

            model.save(filepath=self.config.argsNameModel, save_format='tf', overwrite=True)
            self.config.logger.log_info(f"Model Saved!!!")

    @property
    def _select_transfer_learning_model(self):
        # Modelos disponíveis para Transfer-Learning
        # https://keras.io/api/applications/

        # include_top=False => Excluindo as camadas finais (top layers) da rede, que geralmente são usadas para
        # classificação. Vamos adicionar nossas próprias camadas finais input_shape=(imageDimensionX,
        # imageDimensionY, qtd_canal_color) => Nosso array de imagens tem as dimensões (256, 256, 3) pooling => Modo
        # de pooling opcional para extração de recursos quando include_top for False [none, avg (default), max],
        # passado por parâmetro, mas o default é NONE. weights='imagenet' => Carregamento do modelo inicial com pesos
        # do ImageNet, no qual no treinamento será re-adaptado.
        if self.config.modelSetEnum == ModelSetEnum.ResNet50:
            pretrained_model = tf.keras.applications.ResNet50(include_top=False,
                                                              input_shape=(
                                                                  self.config.imageDimensionX,
                                                                  self.config.imageDimensionY,
                                                                  self.config.channelColors),
                                                              weights='imagenet', pooling='avg')
        elif self.config.modelSetEnum == ModelSetEnum.ResNet101:
            pretrained_model = tf.keras.applications.ResNet101(include_top=False,
                                                               input_shape=(
                                                                   self.config.imageDimensionX,
                                                                   self.config.imageDimensionY,
                                                                   self.config.channelColors),
                                                               weights='imagenet', pooling='avg')
        elif self.config.modelSetEnum == ModelSetEnum.ResNet152:
            pretrained_model = tf.keras.applications.ResNet152(include_top=False,
                                                               input_shape=(
                                                                   self.config.imageDimensionX,
                                                                   self.config.imageDimensionY,
                                                                   self.config.channelColors),
                                                               weights='imagenet', pooling='avg')
        elif self.config.modelSetEnum == ModelSetEnum.ConvNeXtBase:
            pretrained_model = tf.keras.applications.ConvNeXtBase(include_top=False,
                                                                  input_shape=(self.config.imageDimensionX,
                                                                               self.config.imageDimensionY,
                                                                               self.config.channelColors),
                                                                  weights='imagenet', pooling='avg')
        elif self.config.modelSetEnum == ModelSetEnum.ConvNeXtXLarge:
            pretrained_model = tf.keras.applications.ConvNeXtXLarge(include_top=False,
                                                                    input_shape=(self.config.imageDimensionX,
                                                                                 self.config.imageDimensionY,
                                                                                 self.config.channelColors),
                                                                    weights='imagenet', pooling='avg')
        elif self.config.modelSetEnum == ModelSetEnum.EfficientNetB7:
            pretrained_model = tf.keras.applications.EfficientNetB7(include_top=False,
                                                                    input_shape=(self.config.imageDimensionX,
                                                                                 self.config.imageDimensionY,
                                                                                 self.config.channelColors),
                                                                    weights='imagenet', pooling='avg')
        elif self.config.modelSetEnum == ModelSetEnum.EfficientNetV2S:
            pretrained_model = tf.keras.applications.EfficientNetV2S(include_top=False,
                                                                     input_shape=(self.config.imageDimensionX,
                                                                                  self.config.imageDimensionY,
                                                                                  self.config.channelColors),
                                                                     weights='imagenet', pooling='avg')
        elif self.config.modelSetEnum == ModelSetEnum.EfficientNetV2L:
            pretrained_model = tf.keras.applications.EfficientNetV2L(include_top=False,
                                                                     input_shape=(self.config.imageDimensionX,
                                                                                  self.config.imageDimensionY,
                                                                                  self.config.channelColors),
                                                                     weights='imagenet', pooling='avg')

        elif self.config.modelSetEnum == ModelSetEnum.InceptionResNetV2:
            pretrained_model = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                                       input_shape=(self.config.imageDimensionX,
                                                                                    self.config.imageDimensionY,
                                                                                    self.config.channelColors),
                                                                       weights='imagenet', pooling='avg')
        elif self.config.modelSetEnum == ModelSetEnum.DenseNet169:
            pretrained_model = tf.keras.applications.DenseNet169(include_top=False,
                                                                 input_shape=(self.config.imageDimensionX,
                                                                              self.config.imageDimensionY,
                                                                              self.config.channelColors),
                                                                 weights='imagenet', pooling='avg')
        elif self.config.modelSetEnum == ModelSetEnum.VGG19:
            pretrained_model = tf.keras.applications.VGG19(include_top=False,
                                                           input_shape=(
                                                               self.config.imageDimensionX, self.config.imageDimensionY,
                                                               self.config.channelColors),
                                                           weights='imagenet', pooling='avg')
        else:
            raise Exception('Modelo desconhecido')

        pretrained_model.trainable = self.config.argsTrainable
        for layer in pretrained_model.layers:
            layer.trainable = self.config.argsTrainable

        return pretrained_model

# Carregando Modelo
# resnet_model = tf.keras.models.load_model(filepath = self.config.argsNameModel)
# if (self.config.argsDebug):
#     self.config.logger.logInfo(f"{resnet_model.summary()}")
