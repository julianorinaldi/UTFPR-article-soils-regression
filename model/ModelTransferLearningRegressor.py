import os
import shared.infrastructure.helper.DateTimeHelper as helper

import numpy as np
import tensorflow as tf

from dto.ConfigModelDTO import ConfigModelDTO
from dto.FitDTO import FitDTO
from dto.ModelSetEnum import ModelSetEnum
from model.abstract.ModelABCRegressor import ModelABCRegressor
from model.gridsearch.ModelGridSearch import get_config_gridsearch_transfer_learning
from shared.infrastructure.helper.FileHelper import create_file_model


class ModelRegressorTransferLearning(ModelABCRegressor):

    def __init__(self, config: ConfigModelDTO):
        super().__init__(config)

    def get_specialist_model(self, hp):
        # Retora o modelo de TransferLearning selecionado por Enumerador
        self.config.logger.log_info(f"Modelo Selecionado: {self.config.modelSetEnum.name}")
        pretrained_model = self.__select_transfer_learning_model()

        # Adicionando camadas personalizadas no topo do modelo
        layer = pretrained_model.output
        layer = tf.keras.layers.Dense(128, activation='relu')(layer)
        layer = tf.keras.layers.Dropout(0.5)(layer)

        if self.config.argsGridSearch:
            predictions, optimizer = get_config_gridsearch_transfer_learning(hp, layer)
        else:
            layer = tf.keras.layers.Dense(64, activation='relu')(layer)
            predictions = tf.keras.layers.Dense(2, activation='linear')(layer)
            optimizer = tf.keras.optimizers.RMSprop()

        _model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=predictions)
        _model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

        if self.config.argsShowModel:
            self.config.logger.log_info(f"\n{_model.summary()}")

        return _model

    def reshape_two_dimensions(self, x_data):
        return x_data

    def model_fit(self, models, fit_dto: FitDTO):
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_mae', patience=self.config.argsPatience,
            restore_best_weights=True)

        for model in models:
            if not self.config.argsSepared:
                # Padrão sem separação entre validação e treino      
                x_img_data = np.concatenate((fit_dto.x_img_train, fit_dto.x_img_validate), axis=0)
                y_df_train = np.concatenate((fit_dto.y_df_train, fit_dto.y_df_validate), axis=0)
                model.fit(x_img_data, y_df_train, validation_split=0.2, epochs=self.config.argsEpochs,
                          callbacks=[early_stopping])
            else:
                model.fit(fit_dto.x_img_train, fit_dto.y_df_train,
                          validation_data=(fit_dto.x_img_validate, fit_dto.y_df_validate),
                          epochs=self.config.argsEpochs,
                          callbacks=[early_stopping])

            filepath_model = create_file_model(self.config.argsNameModel, "TF")
            model.save(filepath=filepath_model, overwrite=True)
            self.config.logger.log_info(f"Modelo Salvo!!!")

    # Modelos disponíveis para Transfer-Learning
    # https://keras.io/api/applications/

    # include_top=False => Excluindo as camadas finais (top layers) da rede, que geralmente são usadas para
    # classificação. Vamos adicionar nossas próprias camadas finais input_shape=(imageDimensionX,
    # imageDimensionY, qtd_canal_color) => Nosso array de imagens tem as dimensões (256, 256, 3) pooling => Modo
    # de pooling opcional para extração de recursos quando include_top for False [none, avg (default), max],
    # passado por parâmetro, mas o default é NONE. weights='imagenet' => Carregamento do modelo inicial com pesos
    # do ImageNet, no qual no treinamento será re-adaptado.
    def __select_transfer_learning_model(self):
        """
        Seleciona e retorna um modelo de transfer learning pré-treinado com base na configuração.

        Returns:
            tf.keras.Model: O modelo de transfer learning pré-treinado com camadas adicionais para a tarefa de regressão.

        Raises:
            ValueError: Se o modelo especificado não for encontrado no mapeamento.
        """
        model_map = {
            ModelSetEnum.ResNet50: "ResNet50",
            ModelSetEnum.ResNet101: "ResNet101",
            ModelSetEnum.ResNet152: "ResNet152",
            ModelSetEnum.ConvNeXtBase: "ConvNeXtBase",
            ModelSetEnum.ConvNeXtXLarge: "ConvNeXtXLarge",
            ModelSetEnum.EfficientNetB7: "EfficientNetB7",
            ModelSetEnum.EfficientNetV2S: "EfficientNetV2S",
            ModelSetEnum.EfficientNetV2L: "EfficientNetV2L",
            ModelSetEnum.InceptionResNetV2: "InceptionResNetV2",
            ModelSetEnum.DenseNet169: "DenseNet169",
            ModelSetEnum.VGG19: "VGG19",
        }

        model_name = model_map.get(self.config.modelSetEnum)
        if model_name is None:
            raise ValueError(f"Modelo '{self.config.modelSetEnum}' não encontrado no mapeamento.")

        # Use getattr para obter dinamicamente a classe do modelo a partir do nome
        model_class = getattr(tf.keras.applications, model_name)
        input_shape = (self.config.imageDimensionX, self.config.imageDimensionY, self.config.channelColors)

        pretrained_model = model_class(
            include_top=False,
            input_shape=input_shape,
            weights="imagenet",
            pooling="avg",
        )

        # Congelar ou descongelar camadas com base na configuração
        pretrained_model.trainable = self.config.argsTrainable
        for layer in pretrained_model.layers:
            layer.trainable = self.config.argsTrainable

        # Criação do modelo completo
        inputs = tf.keras.Input(shape=input_shape)
        outputs = pretrained_model(inputs, training=pretrained_model.trainable)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

# Carregando Modelo
# resnet_model = tf.keras.models.load_model(filepath = self.config.argsNameModel)
# if (self.config.argsDebug):
#     self.config.logger.logInfo(f"{resnet_model.summary()}")
