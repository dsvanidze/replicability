import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os
ROOT_DIR = os.path.normpath(os.path.join(__file__, *[os.pardir]*2)) # This is your Project Root

def get_data():
    data = {
        "train": pd.read_csv(ROOT_DIR + "/data/csvs/split/train.csv", index_col="index"),
        "validation": pd.read_csv(ROOT_DIR + "/data/csvs/split/validation.csv", index_col="index"),
        "test": pd.read_csv(ROOT_DIR + "/data/csvs/split/test.csv", index_col="index"),
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


def plot_predicted_vs_true(Xs, Ys, model, path=None):
    predicted_values = [np.squeeze(model.predict(X)) for X in Xs]
    true_values = [Y.to_numpy() for Y in Ys]
    titles = ["Training set", "Validation set", "Test set"]
    mses = [model.evaluate(Xs[i], Ys[i], batch_size=Xs[i].shape[0])[0]
            for i in range(3)]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for i in range(3):
        axs[i].scatter(true_values[i], true_values[i],
                       s=10, c="red", alpha=0.3)
        axs[i].scatter(true_values[i], predicted_values[i], s=10, alpha=0.3)
        axs[i].set(xlim=[np.min(true_values[i]) - 0.5, np.max(true_values[i]) + 0.5],
                   ylim=[np.min(true_values[i]) - 0.5,
                         np.max(true_values[i]) + 0.5],
                   xlabel="True values",
                   ylabel="Predicted values")
        axs[i].set_title("{}\nMSE={:.4f}".format(titles[i], mses[i]))
        
    if path:
        plt.savefig(path, dpi=200)
    plt.show()


def plot_true_vs_error(Xs, Ys, model):
    predicted_values = [np.squeeze(model.predict(X)) for X in Xs]
    true_values = [np.squeeze(Y.to_numpy()) for Y in Ys]
    titles = ["Training set", "Validation set", "Test set"]
    mses = [model.evaluate(Xs[i], Ys[i], batch_size=Xs[i].shape[0])[0]
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
        axs[0][i].set_title("{}\nMSE={:.4f}".format(titles[i], mses[i]))

        axs[1][i].scatter(true_values[i], errors[i],
                          c="green", s=10, alpha=0.3)
        axs[1][i].set(xlim=[np.min(true_values[i]) - 0.5, np.max(true_values[i]) + 0.5],
                      ylim=[np.min(errors[i]) - 0.5,
                            np.max(errors[i]) + 0.5],
                      xlabel="True values",
                      ylabel="Errors")

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
