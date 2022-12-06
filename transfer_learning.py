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

    n_samples = []
    svc_accuracies = []
    
    fractions = [n/100 for n in range(5,101,5)]

    for fraction in fractions:
        print(f"Training SVC on {fraction*100:.2f}% of labeled training data...")
        # Fit models
        n = int(fraction * len(train_embeddings))
        X_train = train_embeddings.iloc[:n, 2:]
        y_train = train_embeddings.loc[:, "label_idx"].iloc[:n]
        svm_classifier.fit(X_train, y_train)

        # Predict
        X_test = test_embeddings.iloc[:, 2:]
        y_test = test_embeddings.loc[:, "label_idx"]
        svc_preds = svm_classifier.predict(X_test)

        # Compute accuracies
        svc_accuracy = balanced_accuracy_score(y_test, svc_preds)
        svc_accuracies.append(svc_accuracy)
        n_samples.append(n)

    save_accuracy_plot(svc_accuracies, n_samples, "SVC", figs_path)
