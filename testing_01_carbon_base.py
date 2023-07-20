import numpy as np # Trabalhar com array
import pandas as pd # Trabalhar com análise de dados, importação, etc.
from matplotlib import pyplot as plt # Matplotlib Plot
from tqdm import tqdm # Facilita visualmente a iteração usado no "for"
import tensorflow as tf # Trabalhar com aprendizado de máquinas
import keras # Trabalhar com aprendizado de máquinas
from sklearn.metrics import r2_score # Avaliação das Métricas
import cv2 # Trabalhar com processamento de imagens

from coreProcess import image_processing

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.compat.v1.Session(config=config)
keras.backend.set_session(sess)

# Carregamento do Dataset
df_test = pd.read_csv('dataset/csv/Dataset256x256-Teste.csv')

df_test = df_test.drop(columns={"class", "qtd_mat_org", "nitrog_calc", "amostra", "classe", "tamanho"})

# Definindo o tamanho das imagens
imageDimensionX = 256
imageDimensionY = 256

# Path Dir Teste
dir_name_test = "dataset/images/teste-solo-256x256"

# Separando apenas nomes dos arquivos
test_imagefiles = df_test["arquivo"]

# Removendo coluna arquivo para normalização
df_test = df_test.drop(columns={"arquivo"})

# **********************************************
# **********************************************
# Quantidade de imagens usadas para a rede.
# Foi constatado que depende da quantidade de imagens o Colab quebra por estouro de memória

qtd_imagens = 10000
qtd_canal_color = 3

# Normalização Dataset Teste
test_stats = df_test.describe()
test_stats = test_stats.transpose()
df_test = (df_test - test_stats['mean']) / test_stats['std']

# **********************************************
# **********************************************

# Array com as imagens a serem carregadas de treino
image_list_test = []

for imageFilePath in tqdm(test_imagefiles.tolist()[:qtd_imagens]):
    image_list_test.append(image_processing(dir_name_test, imageFilePath, imageDimensionX, imageDimensionY, qtd_canal_color))
    
# Transformando em array a lista de imagens (Test)
X_test = np.array(image_list_test)
print(f'Shape X_test: {X_test.shape}')

Y_test_carbono = np.array(df_test['teor_carbono'].tolist()[:qtd_imagens])
print(f'Shape Y_test_carbono: {Y_test_carbono.shape}')

# Carregando Modelo
resnet_model = tf.keras.models.load_model('last-model.h5')
print(resnet_model.summary())

##########################
# Trabalhando com R2
prediction = resnet_model.predict(X_test)

r2 = r2_score(Y_test_carbono, prediction)
print(f'R2: {r2}')
##########################

for index in [0,10,20,30,50,100,150,300,500,1000,1400,1800,2000,2500,3000,3500]:
    img_path = f'{dir_name_test}/{test_imagefiles[index]}'
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256, 3))
    x = tf.keras.preprocessing.image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = tf.keras.applications.resnet50.preprocess_input(x)

    x2 = cv2.imread(f'{dir_name_test}/{test_imagefiles[index]}')
    x2 = np.expand_dims(x2, axis=0)

    ResNet50 = resnet_model.predict(x)
    CV2 = resnet_model.predict(x2)
    Real = df_test.teor_carbono[index]

    print(f'ResNet50: {ResNet50.item(0)} => Diferença: {Real - ResNet50.item(0)}')
    print(f'CV2: {CV2.item(0)} => Diferença: {Real - CV2.item(0)}')
    print(f'Image: {test_imagefiles[index]} => {df_test.teor_carbono[index]}')
    print("")

