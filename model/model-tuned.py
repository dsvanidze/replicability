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


HP_NUM_UNITS1 = hp.HParam('num_units1', hp.Discrete([16, 32, 64, 96, 128]))
HP_NUM_UNITS2 = hp.HParam('num_units2', hp.Discrete([16, 32, 64, 96, 128]))
HP_NUM_UNITS3 = hp.HParam('num_units3', hp.Discrete([16, 32, 64, 96, 128]))
HP_NUM_UNITS4 = hp.HParam('num_units4', hp.Discrete([16, 32, 64, 96, 128]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.1, 0.3, 0.5, 0.7, 0.9]))
HP_OPTIMIZER = hp.HParam('optimizer', hp.Discrete(['adam', 'sgd']))

METRIC_ACCURACY = 'mse'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
    hp.hparams_config(
        hparams=[HP_NUM_UNITS1, HP_NUM_UNITS2, HP_NUM_UNITS3,
                 HP_NUM_UNITS4,  HP_DROPOUT, HP_OPTIMIZER],
        metrics=[hp.Metric(METRIC_ACCURACY, display_name='MSE')],
    )


def train_test_model(hparams):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS1], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS2], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS3], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(hparams[HP_NUM_UNITS4], activation=tf.nn.relu),
        tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
        tf.keras.layers.Dense(1, activation="linear"),
    ])
    model.compile(
        optimizer=hparams[HP_OPTIMIZER],
        loss='mean_squared_error',
        metrics=['mse'],
    )

    # Run with 1 epoch to speed things up for demo purposes
    model.fit(X_train, X_train, epochs=10, batch_size=train_total)
    _, mse = model.evaluate(X_test, Y_test)
    return mse


def run(run_dir, hparams):
    with tf.summary.create_file_writer(run_dir).as_default():
        hp.hparams(hparams)  # record the values used in this trial
        mse = train_test_model(hparams)
        tf.summary.scalar(METRIC_ACCURACY, mse, step=1)


session_num = 0

for num_units1 in HP_NUM_UNITS1.domain.values:
    for num_units2 in HP_NUM_UNITS2.domain.values:
        for num_units3 in HP_NUM_UNITS3.domain.values:
            for num_units4 in HP_NUM_UNITS4.domain.values:
                for dropout_rate in HP_DROPOUT.domain.values:
                    for optimizer in HP_OPTIMIZER.domain.values:
                        hparams = {
                            HP_NUM_UNITS1: num_units1,
                            HP_NUM_UNITS2: num_units2,
                            HP_NUM_UNITS3: num_units3,
                            HP_NUM_UNITS4: num_units4,
                            HP_DROPOUT: dropout_rate,
                            HP_OPTIMIZER: optimizer,
                        }
                        run_name = "run-%d" % session_num
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        run('logs/hparam_tuning/' + run_name, hparams)
                        session_num += 1
