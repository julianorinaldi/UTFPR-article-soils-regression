import numpy as np # Trabalhar com array
import pandas as pd # Trabalhar com análise de dados, importação, etc.
from matplotlib import pyplot as plt # Matplotlib Plot
from tqdm import tqdm # Facilita visualmente a iteração usado no "for"
import tensorflow as tf # Trabalhar com aprendizado de máquinas

print(f'Versão do tensorflow: {tf.__version__}')
print(f'Eager: {tf.executing_eagerly()}')
print(f"GPU: {tf.config.list_physical_devices('GPU')}")
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

import keras # Trabalhar com aprendizado de máquinas
from keras.utils import multi_gpu_model
from coreProcess import image_processing

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 1:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.experimental.set_memory_growth(physical_devices[1], True)

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

# Embaralhando o DataFrame
df_train_random = df_train.sample(frac=1, random_state=1)
df_test_random = df_test.sample(frac=1, random_state=1)

# Só funciona em versões mais novas do Pandas
#df_train_random = df_train.sample(frac=1, random_state=1, ignore_index=True)
#df_test_random = df_test.sample(frac=1, random_state=1, ignore_index=True)

# Separando apenas nomes dos arquivos
train_imagefiles = df_train_random["arquivo"]
test_imagefiles = df_test_random["arquivo"]

# Removendo coluna arquivo para normalização
df_train_random = df_train_random.drop(columns={"arquivo"})
df_test_random = df_test_random.drop(columns={"arquivo"})

# Carregando as imagens
print(f'Total de imagens no Dataset Treino: {len(train_imagefiles)}\n')
print(f'Total de imagens no Dataset Teste: {len(test_imagefiles)}\n')

# **********************************************
# **********************************************
# Quantidade de imagens usadas para a rede.
# Foi constatado que depende da quantidade de imagens o Colab quebra por estouro de memória

qtd_imagens = 10000
qtd_canal_color = 3

# **********************************************
# **********************************************

# Normalização Dataset Treinamento
train_stats = df_train_random.describe()
train_stats = train_stats.transpose()
df_train_random_norm = (df_train_random - train_stats['mean']) / train_stats['std']
print(df_train_random_norm.tail())

# Normalização Dataset Teste
test_stats = df_test_random.describe()
test_stats = test_stats.transpose()
df_test_random_norm = (df_test_random - test_stats['mean']) / test_stats['std']
print(df_test_random_norm.tail())


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

Y_train_carbono = np.array(df_train_random_norm['teor_carbono'].tolist()[:qtd_imagens])
print(f'Shape Y_train_carbono: {Y_train_carbono.shape}')

#Y_train_nitrogenio = np.array(df_train['teor_nitrogenio'].tolist()[:qtd_imagens])
#print(f'Shape Y_train_nitrogenio: {Y_train_nitrogenio.shape}')


# Transformando em array a lista de imagens (Test)
X_test = np.array(image_list_test)
print(f'Shape X_test: {X_test.shape}')

Y_test_carbono = np.array(df_test_random_norm['teor_carbono'].tolist()[:qtd_imagens])
print(f'Shape Y_test_carbono: {Y_test_carbono.shape}')

#Y_test_nitrogenio = np.array(df_test['teor_nitrogenio'].tolist()[:qtd_imagens])
#print(f'Shape Y_test_nitrogenio: {Y_test_nitrogenio.shape}')

resnet_model = tf.keras.models.Sequential()

pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                   input_shape=(imageDimensionX, imageDimensionY, qtd_canal_color),
                   pooling='avg', classes=1,
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
parallel_model = multi_gpu_model(resnet_model, gpus=2)

history = parallel_model.fit(X_train, Y_train_carbono, validation_split=0.3, epochs=300, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()

resnet_model.save('last-model.h5')

