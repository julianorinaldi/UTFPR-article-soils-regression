
import tensorflow as tf

from entityModelConfig import ModelConfig
from modelSet import ModelSet

def modelTransferLearningProcess(modelConfig : ModelConfig):

    # Modelos disponíveis para Transfer-Learning
    # https://keras.io/api/applications/
    # ResNet50
    
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
    else:
        raise Exception('Modelo desconhecido')


    # *****************************************************
    # Modelo novo com GlobalAveragePooling2D
    # Parâmetros: sem o camada pooling, drop column teor_nitrogenio, layer.trainable = False, preproc = True
    # *****************************************************
    pretrained_model.trainable = modelConfig.argsTrainable
    for layer in pretrained_model.layers:
        layer.trainable = modelConfig.argsTrainable
    
    return pretrained_model