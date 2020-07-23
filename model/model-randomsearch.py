from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Activation, Lambda, InputLayer
from tensorflow.keras.optimizers import SGD, RMSprop, Adam, Adadelta, Adagrad, Adamax, Nadam, Ftrl
from kerastuner.tuners import RandomSearch
from tensorflow.keras import metrics

import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard, EarlyStopping
from keras import metrics
import pandas as pd
from reproduce import reproduce
import numpy as np
from tensorboard.plugins.hparams import api as hp
import tensorflow as tf
from utils import get_data, plot_predicted_vs_true, custom_r2, custom_adj_r2
from datetime import datetime
from shutil import rmtree
import os
import sys

TRAINING = False

if len(sys.argv) > 1:
    if sys.argv[1] == "train":
        TRAINING = True

begin = datetime.now()
# This set seeds to make the result reproducible
# reproduce(0)


tensorboard = TensorBoard(log_dir="logs/")

# read in data using pandas

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

    for i in range(hp.Int('num_layers', 1, 1)):
        model.add(Dense(units=hp.Int('units_' + str(i), min_value=32,
                                     max_value=4096, step=32),
                        activation='relu',
                        kernel_initializer='he_uniform',
                        bias_initializer='zeros'))
        model.add(BatchNormalization())
        model.add(Dropout(rate=hp.Float('dropout_' + str(i), min_value=0.0,
                                        max_value=0.9, step=0.1)))

    model.add(Dense(1, activation='linear',
                    kernel_initializer='he_uniform',
                    bias_initializer='zeros'))
    model.compile(optimizer=Adam(hp.Choice('learning_rate',
                                           values=[0.005, 0.001, 0.0005, 0.0001, 0.00005, 0.00001])),
                  loss='mean_squared_error',
                  metrics=[metrics.RootMeanSquaredError(), metrics.MeanAbsoluteError(), metrics.MeanAbsolutePercentageError(), metrics.MeanSquaredLogarithmicError()])
    return model


max_trial = 7000

tuner = RandomSearch(
    build_model,
    seed=0,
    objective='val_loss',
    max_trials=max_trial,
    executions_per_trial=1,
    directory='random-search',
    project_name='covid-19-nn')

if TRAINING:
    tuner.search_space_summary()
    # early_stopping_monitor = EarlyStopping(
    #     patience=200, baseline=0.5)
    # Use .values to convert pandas dataframe to numpy array
    # To avoid the Warning -> WARNING:tensorflow:Falling back from v2 loop because of error: Failed to find data adapter that can handle input: <class 'pandas.core.frame.DataFrame'>, <class 'NoneType'>
    tuner.search(X_train.values, Y_train.values,
                 epochs=1000,
                 batch_size=train_total,
                 validation_data=(X_validation.values, Y_validation.values),
                 callbacks=[])

    # all other trial_ids than top 3
    trial_ids_to_remove = [best_trial.trial_id for best_trial in tuner.oracle.get_best_trials(
        num_trials=max_trial)[3:]]

    for trial_id in trial_ids_to_remove:
        rmtree("random-search/covid-19-nn/trial_{}".format(trial_id))
        rmtree("logs/{}".format(trial_id))


print("######## GET BEST MODELS ########")
models = tuner.get_best_models()
evaluation = models[0].evaluate(X_test.values, Y_test.values)
print(models[0].summary())
print(evaluation)
print("Mean Squared Error", evaluation[1])

print(tuner.results_summary(1))

print("R2:", custom_r2(evaluation[0], Y_test.to_numpy()))
print("Adj. R2:", custom_adj_r2(evaluation[0], Y_test.to_numpy(), 16))

print("######## SUMMARY ########")
print("Overall Runtime:", datetime.now() - begin)

if not TRAINING:
    plot_predicted_vs_true(Xs=[X_train, X_validation, X_test],
                           Ys=[Y_train, Y_validation, Y_test],
                           model=models[0])
# # Save the best model
# models[0].save("./models-collection/mlp-model-best-randomsearch-5")


# # Instantiate a model with the best hyperparameters -> makes the model retrainable
# best_hp = tuner.get_best_hyperparameters()[0]
# mode_of_best_hp = tuner.hypermodel.build(best_hp)
# print(mode_of_best_hp.evaluate(X_test, Y_test))
# mode_of_best_hp.fit(X_train, Y_train, batch_size=train_total, validation_data=(X_validation, Y_validation),
#           epochs=1000, verbose=1, callbacks=[])
