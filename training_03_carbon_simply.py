import numpy as np # Trabalhar com array
import pandas as pd # Trabalhar com análise de dados, importação, etc.
from matplotlib import pyplot as plt # Matplotlib Plot
from tqdm import tqdm # Facilita visualmente a iteração usado no "for"
import tensorflow as tf # Trabalhar com aprendizado de máquinas

from coreProcess import image_processing

physical_devices = tf.config.list_physical_devices('GPU')
print("Número de GPUs disponíveis:", len(physical_devices))

# Caso deseje limitar a alocação de memória da GPU (opcional)
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Carregamento do Dataset
df_train = pd.read_csv('dataset/csv/Dataset256x256-Treino.csv')
df_test = pd.read_csv('dataset/csv/Dataset256x256-Teste.csv')

df_train = df_train.drop(columns={"class", "qtd_mat_org", "nitrog_calc", "amostra", "classe", "tamanho"})
df_test = df_test.drop(columns={"class", "qtd_mat_org", "nitrog_calc", "amostra", "classe", "tamanho"})

# Definindo o tamanho das imagens
imageDimensionX = 256
imageDimensionY = 256

# Path Dir Treino
dir_name_train = "dataset/images/treinamento-solo-256x256"
# Path Dir Teste
dir_name_test = "dataset/images/teste-solo-256x256"

# Separando apenas nomes dos arquivos
train_imagefiles = df_train["arquivo"]
test_imagefiles = df_test["arquivo"]

# Removendo coluna arquivo para normalização
df_train = df_train.drop(columns={"arquivo"})
df_test = df_test.drop(columns={"arquivo"})

# **********************************************
# **********************************************
# Quantidade de imagens usadas para a rede.
# Foi constatado que depende da quantidade de imagens o Colab quebra por estouro de memória

qtd_imagens = 500
qtd_canal_color = 3

# Normalização Dataset Treinamento
train_stats = df_train.describe()
train_stats = train_stats.transpose()
df_train = (df_train - train_stats['mean']) / train_stats['std']

# Normalização Dataset Teste
test_stats = df_test.describe()
test_stats = test_stats.transpose()
df_test = (df_test - test_stats['mean']) / test_stats['std']

# **********************************************
# **********************************************

# Array com as imagens a serem carregadas de treino
image_list_train = []

for imageFilePath in tqdm(train_imagefiles.tolist()[:qtd_imagens]):
    image_list_train.append(image_processing(dir_name_train, imageFilePath, imageDimensionX, imageDimensionY, qtd_canal_color))

# Array com as imagens a serem carregadas de treino
image_list_test = []

for imageFilePath in tqdm(test_imagefiles.tolist()[:qtd_imagens]):
    image_list_test.append(image_processing(dir_name_test, imageFilePath, imageDimensionX, imageDimensionY, qtd_canal_color))
    
# Transformando em array a lista de imagens (Treino)
X_train =  np.array(image_list_train)
print(f'Shape X_train: {X_train.shape}')

Y_train_carbono = np.array(df_train['teor_carbono'].tolist()[:qtd_imagens])
print(f'Shape Y_train_carbono: {Y_train_carbono.shape}')

# Transformando em array a lista de imagens (Test)
X_test = np.array(image_list_test)
print(f'Shape X_test: {X_test.shape}')

Y_test_carbono = np.array(df_test['teor_carbono'].tolist()[:qtd_imagens])
print(f'Shape Y_test_carbono: {Y_test_carbono.shape}')

resnet_model = tf.keras.models.Sequential()

pretrained_model= tf.keras.applications.ResNet50(include_top=True,
                   input_shape=(imageDimensionX, imageDimensionY, qtd_canal_color),
                   pooling='avg',
                   weights='imagenet')
for layer in pretrained_model.layers:
        layer.trainable=True

resnet_model.add(pretrained_model)
resnet_model.add(tf.keras.layers.Flatten())
resnet_model.add(tf.keras.layers.Dense(512, activation='relu'))
resnet_model.add(tf.keras.layers.Dense(256, activation='relu'))
resnet_model.add(tf.keras.layers.Dropout(0.5))
resnet_model.add(tf.keras.layers.Dense(1))

print(resnet_model.summary())

opt = tf.keras.optimizers.RMSprop(0.0001)
resnet_model.compile(optimizer=opt,loss='mse',metrics=['mae', 'mse'])
history = resnet_model.fit(X_train, Y_train_carbono, validation_split=0.3, epochs=400, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

resnet_model.save('last-model.h5')

