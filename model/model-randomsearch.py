from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation, Lambda, InputLayer
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch
from tensorflow.keras import metrics

import matplotlib.pyplot as plt
from keras import backend as K

from keras.callbacks import EarlyStopping, TensorBoard
# from keras.layers import Dense, BatchNormalization, Dropout, Activation, Lambda, InputLayer
# from keras.models import Sequential
# from keras import optimizers
from keras import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from reproduce import reproduce
import numpy as np
import time
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf

# This set seeds to make the result reproducible
# reproduce(0)

# NAME = "covid-19-spatial-prediction-MLP-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/")


# def r2_score(y_true, y_pred):
#     from keras import backend as K
#     SS_res = K.sum(K.square(y_true - y_pred))
#     SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
#     return (1 - SS_res/(SS_tot + K.epsilon()))


# def r2_score_adj(y_true, y_pred):
#     # TODO: check if there is better implementation of Adj. R2
#     n = Lambda(lambda x: x[0]/x[1])([K.sum(y_true), K.mean(y_true)])
#     p = K.constant(n_units_of_N_layer)
#     one = K.constant(1)
#     SS_res = K.sum(K.square(y_true - y_pred)) / (n - p - one)
#     SS_tot = K.sum(K.square(y_true - K.mean(y_true))) / (n - one)
#     return (1 - SS_res/(SS_tot + K.epsilon()))

# read in data using pandas

def get_data():
    data = {
        "train": pd.read_csv("data/csvs/split/train.csv", index_col="index"),
        "validation": pd.read_csv("data/csvs/split/validation.csv", index_col="index"),
        "test": pd.read_csv("data/csvs/split/test.csv", index_col="index"),
    }

    for key in data:
        data[key]["longitude"] = data[key]["longitude"] / 180
        data[key]["latitude"] = data[key]["latitude"] / 90
        data[key]["adj_cases"] = np.log(data[key]["adj_cases"] + 1)

    X_train = data["train"].drop("adj_cases", axis=1)
    Y_train = data["train"]["adj_cases"]
    X_validation = data["validation"].drop("adj_cases", axis=1)
    Y_validation = data["validation"]["adj_cases"]
    X_test = data["test"].drop("adj_cases", axis=1)
    Y_test = data["test"]["adj_cases"]

    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test


X_train, Y_train, X_validation, Y_validation, X_test, Y_test = get_data()

# get number of columns in training data
validation_split_rate = 0.2
n_units_of_N_layer = 128
n_cols = X_train.shape[1]
train_total = X_train.shape[0]
batch_size = int(np.floor((1 - validation_split_rate) * train_total))


def build_model(hp):
    model = Sequential()
    model.add(InputLayer(input_shape=(n_cols,)))

    for i in range(hp.Int('num_layers', 2, 10)):
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=16,
                                     max_value=512, step=16),
                        activation='relu',
                        kernel_initializer='he_normal',
                        bias_initializer='zeros'))
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.0,
                                        max_value=0.9, step=0.3)))

    model.add(Dense(1, activation='linear',
                    kernel_initializer='he_normal',
                    bias_initializer='zeros'))
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[
                  1e-2, 1e-3, 1e-4, 1e-5])), loss='mean_squared_error', metrics=[metrics.MeanSquaredError(), metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError(), metrics.MeanAbsolutePercentageError(), metrics.MeanSquaredLogarithmicError(), metrics.Poisson()])
    return model


tuner = RandomSearch(
    build_model,
    seed=0,
    objective='val_mean_squared_error',
    max_trials=50,
    executions_per_trial=1,
    directory='random-search',
    project_name='covid-19-nn')

tuner.search_space_summary()

tuner.search(X_train, Y_train,
             epochs=1000,
             batch_size=train_total,
             validation_data=(X_validation, Y_validation),
             callbacks=[tensorboard])

print("######## GET BEST MODELS ########")
models = tuner.get_best_models()
best_hyperparameters = tuner.get_best_hyperparameters(1)[0]
print(models[0].summary())
print(models[0].evaluate(X_test, Y_test))


def custom_r2(mse, Y):
    n = len(Y)
    ss_res = n*mse
    ss_tot = np.sum(np.square(Y - np.mean(Y)))
    return 1 - (ss_res/ss_tot)


def custom_adj_r2(mse, Y, p):
    n = len(Y)
    standard_term = (n - 1) / (n - p - 1)
    r2 = custom_r2(mse, Y)
    return 1 - (1 - r2) * standard_term


print(custom_r2(0.5159257054328918, Y_test.to_numpy()))
print(custom_adj_r2(0.5159257054328918, Y_test.to_numpy(), 16))

print("######## SUMMARY ########")
# tuner.results_summary(3)
