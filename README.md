# master-degree-soils

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