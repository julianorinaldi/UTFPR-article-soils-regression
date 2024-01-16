# UTFPR Article Soils Regression
Este repositório tem objetivo de estudar e desenvolver modelos de regressões com uso de inteligência artificial, baseados em imagens de solos utilizando aprendizado supervisionado para predizer propriedades químicas dos solos como carbono e nitrogênio.

# Acessando Ambiente
Caso rode local, pule esta etapa... Mas lembre de ter os fontes já baixados.
Aqui é para lembrar de acessar o ambiente UTFPR caso faça o treinamento e teste por lá.
Para auxílio, está criado o arquivo: ```ssh-UTFPR```

```chmod +x ssh-UTFPR```

```./ssh-UTFPR```

# Ambiente VENV
A estrutura do fonte foi criada sobre o VENV, ou seja, crie ou acesse seu ambiente. Procure usar python 3.10 ou superior.
Arquivo: source-env ajudará quando estiver no ambiente UTFPR:
```source source-env```

# Instalando Pacotes com o PIP
Primeiro, vamos atualizar o PIP
```pip install --upgrade pip```

Agora vamos instalar os pacotes que estão no arquivo: ```requirements_UTFPR_py3.10.txt```

```pip install -r requirements_UTFPR_py3.10.txt```

# Como utilizar este repositório?

As pastas principais são:

- raiz do fonte: contém os fontes principais para auxílio dos modelos.
- models: centraliza os fontes de desenvolvimento de cada modelo de regressão.

# Como rodar um modelo?

Considerando que você já entrou no seu ambiente VENV, vamos executar o arquivo ```X-SoilRegression.py```.

O arquivo main ```X-SoilRegression.py``` serve para rodar o treinamento e teste do modelo, contém vários parâmetros de configurações. 

**Para conhecer os parâmetros basta executar o comando ```python3 X-SoilRegression.py -h```.**

```python
usage: X-SoilRegression.py [-h] [-d] [-n NAME] [-p] [-t] [-T] [-i AMOUNT_IMAGE_TRAIN] [-I AMOUNT_IMAGE_TEST] [-P PATIENCE]
                           [-e EPOCHS] [-m MODEL]

options:
  -h, --help            show this help message and exit
  -d, --debug           Para listar os prints de Debug
  -n NAME, --name NAME  Nome do arquivo/diretório de saída do modelo .tf
  -p, --preprocess      Preprocessar imagem 'model.preprocess_input(...)' - [Modelos TransferLearning]
  -t, --trainable       Define se terá as camadas do modelo de transfer-learning treináveis ou não - [Modelos
                        TransferLearning]
  -T, --Test            Define execução apenas para o teste - [Modelos TransferLearning]
  -i AMOUNT_IMAGE_TRAIN, --amount_image_train AMOUNT_IMAGE_TRAIN
                        Quantidade de imagens para treino
  -I AMOUNT_IMAGE_TEST, --amount_image_test AMOUNT_IMAGE_TEST
                        Quantidade de imagens para test
  -P PATIENCE, --patience PATIENCE
                        Quantidade de paciência no early stopping - [Modelos TransferLearning/CNN]
  -e EPOCHS, --epochs EPOCHS
                        Quantidade de épocas para o treino - [Modelos TransferLearning/CNN]
  -m MODEL, --model MODEL
                        Modelo: [0]-ResNet50, [1]-ResNet101, [2]-ResNet152, [10]-ConvNeXtBase, [11]-ConvNeXtXLarge,
                        [20]-EfficientNetB7, [21]-EfficientNetV2S, [22]-EfficientNetV2L, [30]-InceptionResNetV2,
                        [40]-DenseNet169, [50]-VGG19, [100]-CNN, [500]-XGBRegressor, [510]-LinearRegression,
                        [520]-SVMLinearRegression, [521]-SVMRBFRegressor

```

## Exemplo de como rodar um modelo treinamento + teste

O padrão de execução sempre é treinamento + teste, porém você pode escolher apenas testar se desejar, veja o tópico a seguir.

```python3 X-SoilRegression.py -n NomeDoMolo.tf -d -m 40 -p -P 2```

 - -n NAME, --name NAME  Nome do arquivo/diretório de saída do modelo .tf
 - -d, --debug           Para listar os prints de Debug
 - -m MODEL, --model MODEL
            Modelo: [0]-ResNet50, [1]-ResNet101, [2]-ResNet152, [10]-ConvNeXtBase, [11]-ConvNeXtXLarge,
            [20]-EfficientNetB7, [21]-EfficientNetV2S, [22]-EfficientNetV2L, [30]-InceptionResNetV2,
            [40]-DenseNet169, [50]-VGG19, [100]-CNN, [500]-XGBRegressor, [510]-LinearRegression,
            [520]-SVMLinearRegression, [521]-SVMRBFRegressor
 - -p, --preprocess      Preprocessar imagem 'model.preprocess_input(...)' - [Modelos TransferLearning]
 - -P PATIENCE, --patience PATIENCE
            Quantidade de paciência no early stopping - [Modelos TransferLearning/CNN]

## Exemplo de como rodar um modelo apenas de teste

O código abaixo irá procurar o modelo já gerado com o nome **NomeDoMolo.tf**, e executar o teste.

```python3 X-SoilRegression.py -n NomeDoMolo.tf -d -T```

 - -n NAME, --name NAME  Nome do arquivo/diretório de saída do modelo .tf
 - -d, --debug           Para listar os prints de Debug
 - -T, --Test            Define execução apenas para o teste - [Modelos TransferLearning]

# Baixar arquivo last-model.h5 via SSH/SCP
SCP refere-se ao "Secure Copy Protocol" ou "Secure Copy", que é um protocolo de transferência de arquivos seguro. O SCP permite transferir arquivos entre computadores de maneira segura através de uma conexão SSH (Secure Shell).

Qualquer dúvida poderá dar uma olhada nesta documentação: https://linuxize.com/post/how-to-use-scp-command-to-securely-transfer-files/

Diante da sua máquina local (sua máquina), execute o comando abaixo:

```scp -P 2222 jrinaldi@200.134.25.230:"/home/users/jrinaldi/source-py3.10/UTFPR-article-soils-regression/last-model.h5" ~/.```

# Modelos já realizados:
- https://drive.google.com/drive/u/0/folders/1Nt7tv9_Dm31BkixAw7MN5qo6D9V3v3sU

## Modelo Treinado: model-141ep-56.58r2.h5
- training_02_carbon_parallel
- Config:
    - pooling='avg'
    - layer.trainable=True
    - Struct Model:
        - resnet_model.add(tf.keras.layers.Flatten())
        - resnet_model.add(tf.keras.layers.Dense(512, activation='relu'))
        - resnet_model.add(tf.keras.layers.Dense(256, activation='relu'))
        - resnet_model.add(tf.keras.layers.Dropout(0.5))
        - resnet_model.add(tf.keras.layers.Dense(1))
        - opt = tf.keras.optimizers.RMSprop(0.0001)
        - resnet_model.fit(X_train, Y_train_carbono, validation_split=0.3, epochs=300, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)])
- Summary:
    - Stop 141 Epoch
    - R2: 0.5658956203990312

# Novas estratégias para buscar melhores resultados:

## 1) Separar dados de validação do treinamento.
Ou seja, diante do dataset de treino, separar ainda uma parte para validação, garantindo que essas amostras não se encontrem no dataset de treino.

## 2) Estratégia de usar DataArgumentation para melhorar quantidade de imagens.
Implementar diversas formas de DataArgumentarion para aumentar quantidade de imagens para as amostras.

## 3) Método de Ensamble para regressão.
Após fazer a predição do resultado, verificar a média da porcentagem de carbono de todas as imagens de mesmas amostras (de um mesmo grupo, da imagem original), verificar se esta média se aproxima mais do valor real.

## 4) Aprimorar conhecimento sobre regularização L1 (Lasso), L2 (Ridge), Dropout
- Regularização L1 (Lasso):
    A regularização L1 adiciona um termo à função de perda do modelo, proporcional à soma dos valores absolutos dos coeficientes.
    É representada pela fórmula: L1 = λ * Σ|wi|, onde wi são os coeficientes do modelo e λ é o parâmetro de regularização.
    A regularização L1 tende a gerar modelos esparsos, ou seja, alguns coeficientes tornam-se exatamente zero, o que implica na seleção automática de características relevantes.

- Regularização L2 (Ridge):
    A regularização L2 adiciona um termo à função de perda do modelo, proporcional à soma dos quadrados dos coeficientes.
    É representada pela fórmula: L2 = λ * Σwi^2, onde wi são os coeficientes do modelo e λ é o parâmetro de regularização.
    A regularização L2 tende a penalizar coeficientes grandes, ajudando a evitar overfitting e a melhorar a estabilidade do modelo.

- Dropout:
    O Dropout é uma técnica utilizada principalmente em redes neurais, mas pode ser adaptada para outras abordagens de regressão.
    Durante o treinamento, aleatoriamente, alguns neurônios (ou coeficientes, no contexto de regressão) são "desligados" (ou seja, seus valores são definidos como zero).
    Essa técnica ajuda a prevenir o overfitting, pois força o modelo a não depender excessivamente de nenhum conjunto específico de características durante o treinamento.

## 5) Aplicar alguma estratégia de seleção de atributos no modelo.

- Adição de Regularização:
    Adicione técnicas de regularização, como L1 ou L2, às camadas personalizadas para evitar overfitting.

```python
from tensorflow.keras.regularizers import l2
x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
```