import matplotlib.pyplot as plt
from keras import backend as K
from keras.callbacks import EarlyStopping, TensorBoard
from keras.layers import Dense, BatchNormalization, Dropout, Activation, Lambda
from keras.models import Sequential
from keras import optimizers
from keras import metrics
from keras import utils
from keras import losses
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from reproduce import reproduce
import numpy as np
import time
from utils import get_data, plot_predicted_vs_true
# from adabound_tf import AdaBound
from datetime import datetime

begin = datetime.now()
# This set seeds to make the result reproducible
# reproduce(0)

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


X_train, Y_train, X_validation, Y_validation, X_test, Y_test = get_data()

print(X_train.head())
print(Y_train.head())

# get number of columns in training data
validation_split_rate = 0.25
n_units_of_N_layer = 128
n_cols = X_train.shape[1]
train_total = X_train.shape[0]
batch_size = int(np.floor((1 - validation_split_rate) * train_total))

# create model
model = Sequential()

# add model layers
model.add(Dense(2048,
                kernel_initializer='he_uniform',
                bias_initializer='zeros',
                input_shape=(n_cols,)))
# model.add(Dropout(0.9))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(1024,
                kernel_initializer='he_uniform',
                bias_initializer='zeros'))
# model.add(Dropout(0.9))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(512,
                kernel_initializer='he_uniform',
                bias_initializer='zeros'))
# model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(256,
                kernel_initializer='he_uniform',
                bias_initializer='zeros'))
# model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(128,
                kernel_initializer='he_uniform',
                bias_initializer='zeros'))
# model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(4,
                kernel_initializer='he_uniform',
                bias_initializer='zeros'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(1,
                kernel_initializer='he_uniform',
                bias_initializer='zeros'))
# model.add(BatchNormalization())
model.add(Activation("linear"))

# optimizer = optimizers.Adam(learning_rate=0.0001)
optimizer = optimizers.SGD(learning_rate=0.001)
# optimizer = AdaBound(learning_rate=0.001,
#                      final_learning_rate=0.1)

# compile model using mse as a measure of model performance
model.compile(optimizer=optimizer, loss=losses.MeanSquaredError(),
              metrics=[metrics.RootMeanSquaredError()])


# set early stopping monitor so the model stops training when it won"t improve anymore
early_stopping_monitor = EarlyStopping(patience=100)
# train model
history = model.fit(X_train, Y_train, batch_size=train_total, validation_data=(X_validation, Y_validation),
                    epochs=1000, verbose=1, callbacks=[])

# example on how to use our newly trained model on how to make predictions on unseen data (we will pretend our new data is saved in a dataframe called "X_test").
# Y_test_predictions = model.predict(X_test)
# print(np.squeeze(Y_test_predictions))

# newdata = pd.DataFrame(data={"longitude": X_test["longitude"],
#                              "latitude": X_test["latitude"],
#                              "cases": np.squeeze(Y_test_predictions),
#                              "access": X_test["access"],
#                              "longxlat": X_test["longxlat"]})

# print("INVERSED: ", np.squeeze(scaler.inverse_transform(
#     [0, 0, Y_test_predictions, 0, 0])[2]))


results_test = model.evaluate(X_test, Y_test, batch_size=train_total)
print(results_test)


plot_predicted_vs_true(Xs=[X_train, X_validation, X_test],
                       Ys=[Y_train, Y_validation, Y_test],
                       model=model)

# plot_predicted_vs_true(X_validation, Y_validation)
# plot_predicted_vs_true(X_test, Y_test)

# model.save("models-collection/mlp-model-basic.h5")
# print("Saved model to disk")


# # Plot training & validation loss values
# plt.plot(history.history['loss'][::100])
# # plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# # plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

# plt.plot(history.history['val_loss'][::100])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# # plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
print("Overall took:", datetime.now() - begin)
