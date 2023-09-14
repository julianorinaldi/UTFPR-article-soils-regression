# UTFPR Article Soils Regression
Este repositório tem objetivo de estudar e desenvolver modelos de regressões com uso de inteligência artificial, baseados em imagens de solos utilizando aprendizado supervisionado para predizer propriedades químicas dos solos como carbono e nitrogênio.

# Acessando Ambiente
Caso rode local, pule esta etapa... Mas lembre de ter os fontes já baixados.
Aqui é para lembrar de acessar o ambiente UTFPR caso faça o treinamento e teste por lá.
Para auxílio, está criado o arquivo: ```ssh-UTFPR```

```chmod +x ssh-UTFPR```

```./ssh-UTFPR```

# Ambiente VENV
A estrutura do fonte foi criada sobre o VENV, ou seja, crie ou acesso seu ambiente. Procure usar python 3.10 ou superior.
Arquivo: source-env ajudará quando estiver no ambiente UTFPR:
```source source-env```

# Instalando Pacotes com o PIP
Primeiro, vamos atualizar o PIP
```pip install --upgrade pip```

Agora vamos instalar os pacotes que estão no arquivo: ```requirements_UTFPR_py3.10.txt```

```pip install -r requirements_UTFPR_py3.10.txt```

# Baixar arquivo last-model.h5 via SSH/SCP
scp -P 2222 jrinaldi@200.134.25.230:"/home/users/jrinaldi/source/UTFPR-article-soils-regression/last-model.h5" /home/julianorinaldi/Documentos/Mestrado/Modelo-Soils-Regression/.

# Modelos realizados:
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