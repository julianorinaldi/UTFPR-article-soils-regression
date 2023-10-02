import tensorflow as tf  # Trabalhar com aprendizado de máquinas

# Models
from models.modelCNNRegressor import ModelRegressorCNN
from models.modelXGBRegressor import ModelXGBRegressor
from models.modelLinearRegressor import ModelLinearRegressor
from models.modelSVMLinearRegressor import ModelSVMLinearRegressor
from models.modelSVMRBFRegressor import ModelSVMRBFRegressor
from models.modelTransferLearningRegressor import ModelRegressorTransferLearning


from entityModelConfig import ModelConfig
from modelSet import ModelSet


def execute(modelConfig : ModelConfig):
    physical_devices = tf.config.list_physical_devices('GPU')

    # Infos da GPU e Framework
    if (modelConfig.argsDebug):
        print(f'{modelConfig.printPrefix} Amount of GPU Available: {physical_devices}')

    # Estratégia para trabalhar com Multi-GPU
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print(f'{modelConfig.printPrefix} Modelo: {modelConfig.modelSet.name}')
    with strategy.scope():
        if (modelConfig.modelSet == ModelSet.ResNet50 or
            modelConfig.modelSet == ModelSet.ResNet101 or
            modelConfig.modelSet == ModelSet.ResNet152 or
            modelConfig.modelSet == ModelSet.EfficientNetB7 or
            modelConfig.modelSet == ModelSet.EfficientNetV2S or
            modelConfig.modelSet == ModelSet.EfficientNetV2L or
            modelConfig.modelSet == ModelSet.ConvNeXtBase or
            modelConfig.modelSet == ModelSet.ConvNeXtXLarge or
            modelConfig.modelSet == ModelSet.DenseNet169 or
            modelConfig.modelSet == ModelSet.VGG19 or
            modelConfig.modelSet == ModelSet.InceptionResNetV2):
            _model = ModelRegressorTransferLearning(modelConfig)
    
        elif (modelConfig.modelSet == ModelSet.CNN):
            _model = ModelRegressorCNN(modelConfig)
    
        elif (modelConfig.modelSet == ModelSet.XGBRegressor):
            _model = ModelXGBRegressor(modelConfig)

        elif (modelConfig.modelSet == ModelSet.LinearRegression):
            _model = ModelLinearRegressor(modelConfig)

        elif (modelConfig.modelSet == ModelSet.SVMLinearRegression):
            _model = ModelSVMLinearRegressor(modelConfig)

        elif (modelConfig.modelSet == ModelSet.SVMRBFRegressor):
            _model = ModelSVMRBFRegressor(modelConfig)

        else:
            raise Exception('Modelo desconhecido')
        
        _model.train()
        _model.test()
        
        print()
        print(f"{modelConfig.printPrefix} Info parameters: ")
        print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -d (--debug): {modelConfig.argsDebug}")
        print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -n (--name): {modelConfig.argsNameModel}")
        print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -p (--preprocess): {modelConfig.argsPreprocess}")
        print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -t (--trainable): {modelConfig.argsTrainable}")
        print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -T (--Test): {modelConfig.argsOnlyTest}")
        print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -e (--epochs): {modelConfig.argsEpochs}")
        print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -P (--patience): {modelConfig.argsPatience}")
        print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -i (--amount_image_train): {modelConfig.amountImagesTrain}")
        print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -I (--amount_image_test): {modelConfig.amountImagesTest}")
        print(f"{modelConfig.printPrefix}{modelConfig.printPrefix} -m (--model): {modelConfig.modelSet.value} - {modelConfig.modelSet.name}")
        print()