import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping
from keras.layers import Dense, BatchNormalization, Dropout
from keras.models import Sequential
from keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split
from reproduce import reproduce
import numpy as np

# This set seeds to make the result reproducible
reproduce(0)

# read in data using pandas
data = pd.read_csv("data/csvs/cleandata_model.csv")

# data = data.groupby(["longitude", "latitude"], as_index=False).agg(
#     "mean").drop(columns=["Org.ID"])


# data.to_csv("data/csvs/cleandata_model.csv", index=False)
print(data.head())

# create a dataframe with all training data except the target column
X = data.drop(columns=["cases"])

# check that the target variable has been removed
# print(X_train.head())

# create a dataframe with only the target column
Y = data[["cases"]]

# view dataframe
# print(Y_train.head())

# check data has been read in properly
# print(train_df.head())
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

# create model
model = Sequential()

# get number of columns in training data
n_cols = X_train.shape[1]

# add model layers
model.add(Dense(20, activation="relu", input_shape=(n_cols,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(12, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(12, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(8, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(8, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(4, activation="relu"))
model.add(BatchNormalization())
model.add(Dense(1, activation="relu"))

optimizer = Adam(learning_rate=0.000001)

# compile model using mse as a measure of model performance
model.compile(optimizer=optimizer, loss="mean_squared_error",
              metrics=['accuracy'])


# set early stopping monitor so the model stops training when it won"t improve anymore
early_stopping_monitor = EarlyStopping(patience=200)
# train model
history = model.fit(X_train, Y_train, validation_split=0.2,
                    epochs=10000, callbacks=[early_stopping_monitor])

# example on how to use our newly trained model on how to make predictions on unseen data (we will pretend our new data is saved in a dataframe called "X_test").
# Y_test_predictions = model.predict(X_test)
# print(Y_test_predictions)

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
