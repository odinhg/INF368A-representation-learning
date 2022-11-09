import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
from os.path import isfile
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from configfile import *
from utilities import save_accuracy_plot

if __name__ == "__main__":
    if not isfile(embeddings_file_unseen):
        exit("Embeddings not found. Please run embed.py first!")
    
    # Load saved embeddings of the data with unseen classes and split
    df = pd.read_pickle(embeddings_file_unseen)
    train_embeddings, test_embeddings = train_test_split(df, test_size=0.35, shuffle=True, random_state=0)

    svm_classifier = make_pipeline(StandardScaler(), SVC(gamma="auto"))
    linear_classifier = make_pipeline(StandardScaler(), SGDClassifier(loss="hinge"))
    k = 10
    knn_classifier = KNeighborsClassifier(n_neighbors=k)

    n_samples = []
    svc_accuracies = []
    linear_accuracies = []
    knn_accuracies = []

    sample_range = chain(range(30, 100, 10), range(100, 1000, 100), range(1000, len(train_embeddings), 200))
    #for ratio in tqdm([x/20 for x in range(1,20)]):
    for n in tqdm(list(sample_range)):
        # Fit models
        X_train = train_embeddings.iloc[:n, 2:]
        y_train = train_embeddings.loc[:, "label_idx"].iloc[:n]
        svm_classifier.fit(X_train, y_train)
        linear_classifier.fit(X_train, y_train)
        knn_classifier.fit(X_train, y_train)

        # Predict
        X_test = test_embeddings.iloc[:, 2:]
        y_test = test_embeddings.loc[:, "label_idx"]
        svc_preds = svm_classifier.predict(X_test)
        linear_preds = linear_classifier.predict(X_test)
        knn_preds = knn_classifier.predict(X_test)

        # Compute accuracies
        svc_accuracy = balanced_accuracy_score(y_test, svc_preds)
        linear_accuracy = balanced_accuracy_score(y_test, linear_preds)
        knn_accuracy = balanced_accuracy_score(y_test, knn_preds)

        svc_accuracies.append(svc_accuracy)
        linear_accuracies.append(linear_accuracy)
        knn_accuracies.append(knn_accuracy)
        n_samples.append(n)

    save_accuracy_plot(svc_accuracies, n_samples, "SVC", figs_path)
    save_accuracy_plot(linear_accuracies, n_samples, "Linear", figs_path)
    save_accuracy_plot(knn_accuracies, n_samples, "kNN", figs_path)
