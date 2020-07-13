import pandas as pd
import numpy as np

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
    Y_train = data["train"][["adj_cases"]]
    X_validation = data["validation"].drop("adj_cases", axis=1)
    Y_validation = data["validation"][["adj_cases"]]
    X_test = data["test"].drop("adj_cases", axis=1)
    Y_test = data["test"][["adj_cases"]]

    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test
