import tensorflow as tf

def get_config_gridsearch_transfer_learning(hp, layer):
    hp_dense1 = hp.Float('dense1', min_value=32, max_value=256, step=32)
    layer = tf.keras.layers.Dense(hp_dense1, activation='relu')(layer)
    hp_dropout1 = hp.Float('dropuot_rate1', min_value=0.3, max_value=0.5, step=0.1)
    layer = tf.keras.layers.Dropout(hp_dropout1)(layer)

    predictions = tf.keras.layers.Dense(2, activation=hp.Choice('activation', values=['linear']))(layer)

    optimizer = tf.keras.optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[0.0001]))

    return predictions, optimizer

