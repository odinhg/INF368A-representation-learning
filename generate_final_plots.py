import pandas as pd
import matplotlib.pyplot as plt
from configfile import models
from os.path import join, isfile

val_accuracies = pd.DataFrame()
test_accuracies = pd.DataFrame()

for model in models.keys():
    figspath = join("figs", model)
    train_history_filename = join(figspath, "train_history.pkl")
    accuracies_filename = join(figspath, "svc_accuracies.pkl")
    if isfile(train_history_filename):
        val_accuracies[model] = pd.read_pickle(train_history_filename)["val_accuracy"]
    if isfile(accuracies_filename):
        test_accuracies[model] = pd.read_pickle(accuracies_filename)["accuracy"]

val_plot = val_accuracies.plot(title="Validation accuracy (Seen data)", xlabel="Steps", ylabel="Accuracy [%]").get_figure()
val_plot.savefig("figs/val_acc_all.png")

test_plot = val_accuracies.plot(title="Test accuracy (Unseen data)", xlabel="Training dataset size", ylabel="Accuracy [%]").get_figure()
test_plot.savefig("figs/test_acc_all.png")
