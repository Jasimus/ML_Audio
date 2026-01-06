from tensorflow import keras
from keras.optimizers import Adam
from numpy import inf
import os


def build_model(input_steps, output_steps, n_features, n_cells=4, lr=0.01, hidden_layers=1, hidden_activation="relu"):
    optimizer = Adam(learning_rate=lr)

    input_l = keras.layers.Input(shape=(input_steps, n_features))
    x = input_l

    for _ in range(hidden_layers):
        x = keras.layers.SimpleRNN(n_cells, activation=hidden_activation, return_sequences=True)(x)

    output_l = keras.layers.Dense(n_features, activation="relu")(x)

    output_l = output_l[:, -output_steps:, :]
    model = keras.Model(inputs=input_l, outputs=output_l)
    model.compile(optimizer=optimizer,
                 loss='mse',
                 metrics=['mse'])

    return model


def take_model(path, model_name):
    ckp = None
    ckp_list = os.listdir(path)

    for c in ckp_list:
        if c == f'{model_name}.keras':
            ckp = c
            break
    
    return ckp

