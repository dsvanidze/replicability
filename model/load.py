from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import load_model
from reproduce import reproduce
import pandas as pd

# This set seeds to make the result reproducible
reproduce(0)

# load model
model = load_model('basic-model.h5')
# summarize model.
model.summary()
# read in data using pandas
data = pd.read_csv("data/csvs/cleandata-basic-model.csv")

scaler = StandardScaler()
data[data.columns] = scaler.fit_transform(data[data.columns])
print(data.head())

# create a dataframe with all training data except the target column
X = data.drop(columns=["cases"])

# create a dataframe with only the target column
Y = data[["cases"]]

# check data has been read in properly
_, X_test, _, Y_test = train_test_split(X, Y, test_size=0.2)


# evaluate the model
score = model.evaluate(X_test, Y_test, verbose=0)
print("Model evaluation: ", score, "MSE")
