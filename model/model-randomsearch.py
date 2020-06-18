from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

import matplotlib.pyplot as plt
from keras import backend as K

from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, BatchNormalization, Dropout, Activation, Lambda
from keras.models import Sequential
from keras import optimizers
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
reproduce(0)

NAME = "covid-19-spatial-prediction-MLP-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


def r2_score(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res/(SS_tot + K.epsilon()))


def r2_score_adj(y_true, y_pred):
    # TODO: check if there is better implementation of Adj. R2
    n = Lambda(lambda x: x[0]/x[1])([K.sum(y_true), K.mean(y_true)])
    p = K.constant(n_units_of_N_layer)
    one = K.constant(1)
    SS_res = K.sum(K.square(y_true - y_pred)) / (n - p - one)
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) / (n - one)
    return (1 - SS_res/(SS_tot + K.epsilon()))


# read in data using pandas
raw_data = pd.read_csv("data/csvs/data.v1.1.csv")
print(len(raw_data))

data = raw_data.drop(raw_data[raw_data.sampling == 0].index).dropna().drop(
    columns=['id', 'pop', 'logpop', 'nbHF', 'adjpop', 'sampling', 'total_cases', 'elevation', 'aridity', 'irrigation'])
print(data.head())

data['adj_cases'] = np.log(data[["adj_cases"]] + 1)

print(data['adj_cases'])


# create a dataframe with all training data except the target column
X = data.drop(columns=["adj_cases"])

# create a dataframe with only the target column
Y = data[["adj_cases"]]

# check data has been read in properly
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# data[data.columns, -"cases"] = scaler.fit_transform(data[data.columns, -"cases"])
print(X_train)
print(X_test)

# get number of columns in training data
validation_split_rate = 0.2
n_units_of_N_layer = 128
n_cols = X_train.shape[1]
train_total = X_train.shape[0]
batch_size = int(np.floor((1 - validation_split_rate) * train_total))


def build_model(hp):
    model = keras.Sequential()
    model.add(layers.Dense(units=hp.Int('units1', min_value=32,
                                        max_value=512, step=32), activation='relu'))
    model.add(layers.Dropout(rate=hp.Float('dropout1', min_value=0.0,
                                           max_value=0.99, step=0.2)))

    model.add(layers.Dense(units=hp.Int('units2', min_value=32,
                                        max_value=512, step=32), activation='relu'))
    model.add(layers.Dropout(rate=hp.Float('dropout2', min_value=0.0,
                                           max_value=0.99, step=0.2)))

    model.add(layers.Dense(units=hp.Int('units3', min_value=32,
                                        max_value=512, step=32), activation='relu'))
    model.add(layers.Dropout(rate=hp.Float('dropout3', min_value=0.0,
                                           max_value=0.99, step=0.2)))

    model.add(layers.Dense(1, activation='linear'))
    model.compile(
        optimizer=keras.optimizers.Adam(
            hp.Choice('learning_rate',
                      values=[1e-2, 1e-3, 1e-4])),
        loss='mean_squared_error',
        metrics=['mse'])
    return model


tuner = RandomSearch(
    build_model,
    seed=0,
    objective='val_mse',
    max_trials=5,
    executions_per_trial=3,
    directory='random-search',
    project_name='covid-19-nn')

tuner.search_space_summary()

tuner.search(X_train, Y_train,
             epochs=1000,
             batch_size=train_total,
             validation_data=(X_test, Y_test))

print("######## GET BEST MODELS ########")
models = tuner.get_best_models(num_models=2)
models.save
print(models)

print("######## SUMMARY ########")
tuner.results_summary()
