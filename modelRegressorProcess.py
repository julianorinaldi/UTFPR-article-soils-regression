
from entityModelConfig import ModelConfig
from datasetProcess import dataset_process
from imageProcess import image_load, image_convert_array
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score  # Avaliação das Métricas

class ModelRegressorProcess:
    def __init__(self, modelConfig : ModelConfig):
        self.modelConfig = modelConfig
        self.model = None

    def _load_images(self, modelConfig : ModelConfig):
        df, imageNamesList = dataset_process(modelConfig)

        # Quantidade de imagens usadas para a rede.
        qtd_imagens = len(df)
        if (modelConfig.argsDebug):
            print(f'{modelConfig.printPrefix} Preprocess: {modelConfig.argsPreprocess}')

        # Array com as imagens a serem carregadas de treino
        imageArray = image_load(modelConfig, imageNamesList, qtd_imagens)

        X_, Y_carbono = image_convert_array(modelConfig, imageArray, df, qtd_imagens)
        
        return X_, Y_carbono
        
    def train(self):
        self.modelConfig.setDirBaseImg('dataset/images/treinamento-solo-256x256')
        self.modelConfig.setPathCSV('dataset/csv/Dataset256x256-Treino.csv')
        
        X_, Y_carbono = self._load_images(self.modelConfig)
        
        # Flatten das imagens
        X_ = X_.reshape(X_.shape[0], -1)  # RandomForestRegressor aceita apenas 2 dimensões.

        # Criar o modelo RandomForestRegressor
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Treinar o modelo
        self.model.fit(X_, Y_carbono)
        
    def test(self):
        # Agora entra o Test
        self.modelConfig.setDirBaseImg('dataset/images/teste-solo-256x256')
        self.modelConfig.setPathCSV('dataset/csv/Dataset256x256-Teste.csv')
        
        X_, Y_carbono = self._load_images(self.modelConfig)
        
        # Fazendo a predição sobre os dados de teste
        prediction = self.model.predict(X_)

        # Avaliando com R2
        r2 = r2_score(Y_carbono, prediction)
        print()
        print(f'====================================================')
        print(f'====================================================')
        print(f'=========>>>>> R2: {r2} <<<<<=========')
        print(f'====================================================')
        print(f'====================================================')