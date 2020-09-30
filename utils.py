import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

import os
ROOT_DIR = os.path.normpath(os.path.join(__file__, *[os.pardir]*1)) # This is the Project Root path

def get_data():
    data = {
        "train": pd.read_csv(ROOT_DIR + "/data/csvs/split/train.csv", index_col="index"),
        "validation": pd.read_csv(ROOT_DIR + "/data/csvs/split/validation.csv", index_col="index"),
        "test": pd.read_csv(ROOT_DIR + "/data/csvs/split/test.csv", index_col="index"),
    }

    for key in data:
        data[key]["LONG"] = data[key]["LONG"] / 180
        data[key]["LAT"] = data[key]["LAT"] / 90
        data[key]["IR"] = np.log(data[key]["IR"] + 1)

    X_train = data["train"].drop("IR", axis=1)
    Y_train = data["train"][["IR"]]
    X_validation = data["validation"].drop("IR", axis=1)
    Y_validation = data["validation"][["IR"]]
    X_test = data["test"].drop("IR", axis=1)
    Y_test = data["test"][["IR"]]

    return X_train, Y_train, X_validation, Y_validation, X_test, Y_test


def plot_predicted_vs_true(Xs, Ys, predicted_set_values, path=None):
    predicted_values = predicted_set_values
    true_values = [Y.values for Y in Ys]
    titles = ["Training set", "Validation set", "Test set"]
    rmses = [np.sqrt(mean_squared_error(y_true=Ys[i], 
                                        y_pred=predicted_set_values[i]))
            for i in range(3)]

    fig, axs = plt.subplots(1, 3, figsize=(24, 6))
    for i in range(3):
        axs[i].scatter(true_values[i], true_values[i],
                       s=10, c="red", alpha=0.3)
        axs[i].scatter(true_values[i], predicted_values[i], s=10, alpha=0.3)
        axs[i].set(xlim=[np.min(true_values[i]) - 0.5, np.max(true_values[i]) + 0.6],
                   ylim=[np.min(true_values[i]) - 0.5,
                         np.max(true_values[i]) + 0.6])
        
        axs[i].tick_params(labelsize=20)
        axs[i].set_xlabel(xlabel="True values", fontsize=20, labelpad=20)
        axs[i].set_ylabel(ylabel="Predicted values", fontsize=20, labelpad=20)
        axs[i].set_title("{}\nRMSE={:.4f}".format(titles[i], rmses[i]),
                         fontdict={"fontsize": 25}, 
                         pad=20)
        
    if path:
        plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.show()


def plot_true_vs_error(Xs, Ys, model):
    predicted_values = [np.squeeze(model.predict(X)) for X in Xs]
    true_values = [np.squeeze(Y.to_numpy()) for Y in Ys]
    titles = ["Training set", "Validation set", "Test set"]
    rmses = [np.sqrt(model.evaluate(Xs[i], Ys[i], verbose=0)[0])
            for i in range(3)]
    errors = [true_values[i] - predicted_values[i] for i in range(3)]

    fig, axs = plt.subplots(2, 3, figsize=(16, 8))
    for i in range(3):
        axs[0][i].scatter(true_values[i], true_values[i],
                          s=10, c="red", alpha=0.3)
        axs[0][i].scatter(true_values[i], predicted_values[i], s=10, alpha=0.3)
        axs[0][i].set(xlim=[np.min(true_values[i]) - 0.5, np.max(true_values[i]) + 0.5],
                      ylim=[np.min(true_values[i]) - 0.5,
                            np.max(true_values[i]) + 0.5],
                      xlabel="True values",
                      ylabel="Predicted values")
        axs[0][i].tick_params(labelsize=20)
        axs[0][i].set_title("{}\nRMSE={:.4f}".format(titles[i], rmses[i]))

        axs[1][i].scatter(true_values[i], errors[i],
                          c="green", s=10, alpha=0.3)
        axs[1][i].set(xlim=[np.min(true_values[i]) - 0.5, np.max(true_values[i]) + 0.5],
                      ylim=[np.min(errors[i]) - 0.5,
                            np.max(errors[i]) + 0.5],
                      xlabel="True values",
                      ylabel="Errors")
        axs[1][i].tick_params(labelsize=20)

    plt.show()


def plot_error_histograms(Xs, Ys, model):
    import seaborn as sns
    predicted_values = [np.squeeze(model.predict(X)) for X in Xs]
    true_values = [np.squeeze(Y.to_numpy()) for Y in Ys]
    titles = ["Training set", "Validation set", "Test set"]
    errors = [true_values[i] - predicted_values[i] for i in range(3)]

    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    for i in range(3):
        sns.distplot(errors[i], ax=axs[i], color="dodgerblue")
        axs[i].set_title("{}\nErrors".format(titles[i]))

    plt.show()


def custom_r2(Y_true, Y_predictions):
    n = len(Y_true.values)
    mse = mean_squared_error(y_true=Y_true, y_pred=Y_predictions)
    ss_res = n*mse
    ss_tot = np.sum(np.square(Y_true.values - np.mean(Y_true.values)))
    return 1 - (ss_res/ss_tot)


def custom_adj_r2(Y_true, Y_predictions, p):
    n = len(Y_true.values)
    standard_term = (n - 1) / (n - p - 1)
    mse = mean_squared_error(y_true=Y_true.values, y_pred=Y_predictions)
    r2 = custom_r2(Y_true, Y_predictions)
    return 1 - (1 - r2) * standard_term
