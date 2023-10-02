from entityModelConfig import ModelConfig
from datasetProcess import dataset_process
from imageProcess import image_load, image_convert_array
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

class ModelLinearRegression:
    def __init__(self, modelConfig : ModelConfig):
        self.modelConfig = modelConfig
        self.model = None

    def _load_images(self, modelConfig : ModelConfig, qtdImagens : int):
        df, imageNamesList = dataset_process(modelConfig)

        # Quantidade de imagens usadas para a rede.
        qtd_imagens = len(df)
        if (qtd_imagens > qtdImagens) and (qtdImagens > 0):
            qtd_imagens = qtdImagens

        # Array com as imagens a serem carregadas de treino
        imageArray = image_load(modelConfig, imageNamesList, qtd_imagens)

        X_, Y_carbono = image_convert_array(modelConfig, imageArray, df, qtd_imagens)
        
        return X_, Y_carbono
        
    def train(self):
        self.modelConfig.setDirBaseImg('dataset/images/treinamento-solo-256x256')
        self.modelConfig.setPathCSV('dataset/csv/Dataset256x256-Treino.csv')
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Carregando imagens para o treino')
        X_, Y_carbono = self._load_images(self.modelConfig, qtdImagens=self.modelConfig.amountImagesTrain)
        
        # Flatten das imagens
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Fazendo reshape')
        
        # Aceita apenas 2 dimensões.
        X_ = X_.reshape(X_.shape[0], -1)  
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Novo shape de X_: {X_.shape}')
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Criando modelo: {self.modelConfig.modelSet.name}')
        
        self.model = LinearRegression()

        # Treinar o modelo
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Iniciando o treino')
        self.model.fit(X_, Y_carbono)
        
    def test(self):
        # Agora entra o Test
        self.modelConfig.setDirBaseImg('dataset/images/teste-solo-256x256')
        self.modelConfig.setPathCSV('dataset/csv/Dataset256x256-Teste.csv')
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Carregando imagens para o teste')
            
        X_, Y_carbono = self._load_images(self.modelConfig, qtdImagens=self.modelConfig.amountImagesTest)
        
        # Aceita apenas 2 dimensões.
        X_ = X_.reshape(X_.shape[0], -1)  
        
        if (self.modelConfig.argsDebug):
            print(f'{self.modelConfig.printPrefix} Iniciando predição...')
        
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