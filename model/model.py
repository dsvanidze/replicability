import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.layers import Dense, BatchNormalization, Dropout, Activation
from keras.models import Sequential
from keras import optimizers
from keras import metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from reproduce import reproduce
import numpy as np

# This set seeds to make the result reproducible
reproduce(0)

# read in data using pandas
data = pd.read_csv("data/csvs/cleandata-basic-model.csv")

# create a dataframe with all training data except the target column
X = data.drop(columns=["cases"])

# create a dataframe with only the target column
Y = data[["cases"]]

# check data has been read in properly
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
# data[data.columns, -"cases"] = scaler.fit_transform(data[data.columns, -"cases"])
print(X_train)
print(X_test)

# create model
model = Sequential()

# get number of columns in training data
n_cols = X_train.shape[1]

# add model layers
model.add(Dense(20,
                kernel_initializer='he_normal',
                bias_initializer='zeros',
                input_shape=(n_cols,)))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(20,
                kernel_initializer='he_normal',
                bias_initializer='zeros'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(16,
                kernel_initializer='he_normal',
                bias_initializer='zeros'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(8,
                kernel_initializer='he_normal',
                bias_initializer='zeros'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(8,
                kernel_initializer='he_normal',
                bias_initializer='zeros'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(Activation("relu"))

model.add(Dense(1,
                kernel_initializer='he_normal',
                bias_initializer='zeros'))
model.add(BatchNormalization())
model.add(Activation("linear"))

optimizer = optimizers.Adam(learning_rate=0.0002)

# compile model using mse as a measure of model performance
model.compile(optimizer=optimizer, loss="mean_squared_error")


# set early stopping monitor so the model stops training when it won"t improve anymore
# early_stopping_monitor = EarlyStopping(patience=5000)
# train model
history = model.fit(X_train, Y_train, batch_size=512, validation_split=0.2,
                    epochs=10000, verbose=0, callbacks=[])

# example on how to use our newly trained model on how to make predictions on unseen data (we will pretend our new data is saved in a dataframe called "X_test").
#Y_test_predictions = model.predict(X_test)
#print(np.squeeze(Y_test_predictions))

# newdata = pd.DataFrame(data={"longitude": X_test["longitude"],
#                              "latitude": X_test["latitude"],
#                              "cases": np.squeeze(Y_test_predictions),
#                              "access": X_test["access"],
#                              "longxlat": X_test["longxlat"]})

# print("INVERSED: ", np.squeeze(scaler.inverse_transform(
#     [0, 0, Y_test_predictions, 0, 0])[2]))
results_train = model.evaluate(X_train, Y_train, batch_size=512)
print(results_train)


results_test = model.evaluate(X_test, Y_test, batch_size=512)
print(results_test)

model.save("models-collection/basic-model.h5")
print("Saved model to disk")


# Plot training & validation loss values
plt.plot(history.history['loss'][::100])
# plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['val_loss'][::100])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
