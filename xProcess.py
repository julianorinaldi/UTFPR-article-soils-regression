import tensorflow as tf  # Trabalhar com aprendizado de máquinas

# Models
from models.modelCNNRegressor import ModelRegressorCNN
from models.modelXGBRegressor import ModelXGBRegressor
from models.modelLinearRegressor import ModelLinearRegressor
from models.modelSVMLinearRegressor import ModelSVMLinearRegressor
from models.modelSVMRBFRegressor import ModelSVMRBFRegressor

from trainingCarbon import TrainingCarbon
from testingCarbon import TestCarbon
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
        if (modelConfig.modelSet == ModelSet.XGBRegressor):
            modelXGBRegressor = ModelXGBRegressor(modelConfig)
            modelXGBRegressor.train()
            modelXGBRegressor.test()
        elif (modelConfig.modelSet == ModelSet.CNN):
            modelRegressorCNN = ModelRegressorCNN(modelConfig)
            modelRegressorCNN.train()
            modelRegressorCNN.test()
        elif (modelConfig.modelSet == ModelSet.LinearRegression):
            modelLinearRegressor = ModelLinearRegressor(modelConfig)
            modelLinearRegressor.train()
            modelLinearRegressor.test()
        elif (modelConfig.modelSet == ModelSet.SVMLinearRegression):
            modelSVMLinearRegressor = ModelSVMLinearRegressor(modelConfig)
            modelSVMLinearRegressor.train()
            modelSVMLinearRegressor.test()
        elif (modelConfig.modelSet == ModelSet.SVMRBFRegressor):
            modelSVMRBFRegressor = ModelSVMRBFRegressor(modelConfig)
            modelSVMRBFRegressor.train()
            modelSVMRBFRegressor.test()
        else:
            # Aqui entra os Modelos de TransferLearning
            if (not modelConfig.argsOnlyTest):
                print()
                print(f'{modelConfig.printPrefix}')
                print(f'{modelConfig.printPrefix} Iniciando o Treino')
                print(f'{modelConfig.printPrefix}')
                modelConfig.setDirBaseImg('dataset/images/treinamento-solo-256x256')
                modelConfig.setPathCSV('dataset/csv/Dataset256x256-Treino.csv')
                training = TrainingCarbon(modelConfig)
                training.train()
            else:
                print()
                print(f'{modelConfig.printPrefix}')
                print(f'{modelConfig.printPrefix} Somente execução por Teste')
                print(f'{modelConfig.printPrefix}')
            
            print()
            print(f'{modelConfig.printPrefix}')
            print(f'{modelConfig.printPrefix} Iniciando o Teste')
            print(f'{modelConfig.printPrefix}')
            modelConfig.setDirBaseImg('dataset/images/teste-solo-256x256')
            modelConfig.setPathCSV('dataset/csv/Dataset256x256-Teste.csv')
            testCarbon = TestCarbon(modelConfig)
            testCarbon.test()
        
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