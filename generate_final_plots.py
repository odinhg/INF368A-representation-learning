import pandas as pd
import matplotlib.pyplot as plt
from configfile import models
from os.path import join, isfile

val_accuracies = pd.DataFrame()
test_accuracies = pd.DataFrame()
test_accuracies["Dataset size"] = [n for n in range(5,101,5)]

for model in models.keys():
    figspath = join("figs", model)
    train_history_filename = join(figspath, "train_history.pkl")
    accuracies_filename = join(figspath, "svc_accuracies.pkl")
    if isfile(train_history_filename):
        val_accuracies[model] = pd.read_pickle(train_history_filename)["val_accuracy"].rolling(5, min_periods=1).mean()
    if isfile(accuracies_filename):
        test_accuracies[model] = pd.read_pickle(accuracies_filename)["accuracy"]

val_plot = val_accuracies.plot(title="Validation accuracy (SVC on small validation dataset)", xlabel="Steps", ylabel="Accuracy").get_figure()
val_plot.savefig("figs/val_acc_all.png")

test_plot = test_accuracies.plot(x="Dataset size", title="Test accuracy (SVC on unseen data)", xlabel="Training data used [%]", ylabel="Accuracy").get_figure()
test_plot.savefig("figs/test_acc_all.png")
