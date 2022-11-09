import torch
import numpy as np
import pandas as pd
from os.path import join
from tqdm import tqdm
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import random
from torchvision import transforms

def save_train_plot(filename, train_history):
    # Plot losses and accuracies from training
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    axes[0].plot(train_history["train_loss"], 'b', label="Train")
    axes[0].plot(train_history["val_loss"], 'g', label="Val")
    axes[1].plot(train_history["train_accuracy"], 'b', label="Train")
    axes[1].plot(train_history["val_accuracy"], 'g', label="Val")
    axes[0].title.set_text('Loss')
    axes[1].title.set_text('Accuracy')
    axes[0].legend(loc="upper right")
    axes[1].legend(loc="upper right")
    fig.tight_layout()
    plt.savefig(filename)

def save_loss_plot(filename, train_history):
    # Plot training and validation loss (for triplet loss model)
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    axes[0].plot(train_history["train_loss"], 'b', label="Train")
    axes[1].plot(train_history["val_loss"], 'g', label="Val")
    axes[0].title.set_text('Training Loss')
    axes[1].title.set_text('Validation Loss')
    fig.tight_layout()
    plt.savefig(filename)


def compute_average_distances(classes):
    # Compute average Euclidean and angular distances between classes
    # classes: list of pandas dataframes with embeddings for each class
    avg_euclidean_distances = np.zeros((len(classes), len(classes)))
    avg_angular_distances = np.zeros((len(classes), len(classes)))
    for i in tqdm(range(len(classes))):
        for j in range(len(classes)):
            avg_euclidean_distance = np.mean(cdist(classes[i], classes[j], metric="euclidean"))
            avg_euclidean_distances[i,j] = avg_euclidean_distance
            avg_angular_distance = np.mean(cdist(classes[i], classes[j], metric="cosine"))
            avg_angular_distances[i,j] = avg_angular_distance
    return (avg_euclidean_distances, avg_angular_distances)

def save_distance_figure(distances, class_names, filename):
    # Save average distances to distance matrix plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(distances, interpolation="nearest")
    fig.colorbar(cax)
    #ax.set_xticklabels([''] + class_names, rotation=45, ha="left")
    #ax.set_yticklabels([''] + class_names)
    fig.tight_layout()
    plt.savefig(filename)
    plt.cla()

def sample_df(df, n=100):
    # Sample n rows randomly from a dataframe
    if n > df.shape[0]:
        n = df.shape[0]
    return df.sample(n, random_state=420)

def save_embeddings(backbone, class_idx, dataloader, filename, device):
    # Compute and save embeddings to pickled dataframes
    embeddings = []
    backbone.eval()
    with torch.no_grad():
        for data in tqdm(dataloader):
            images, labels, indicies = data[0].to(device), data[1].to(device), data[2].to(device)
            activations_second_last_layer = backbone(images) #We don't care about predictions, just embeddings
            embeddings += [[int(class_idx[label]), int(index)]  + activation for activation, label, index in zip(activations_second_last_layer.cpu().detach().tolist(), labels.cpu().detach().tolist(), indicies.cpu().detach().tolist())]
    df = pd.DataFrame(data=embeddings)
    df.columns = ["label_idx", "image_idx"] + [f"X{i}" for i in range(1, df.shape[1] - 1)]
    df.to_pickle(filename)
    print(f"Dataframe ({df.shape[0]} x {df.shape[1]}) saved to {filename}")

class EarlyStopper():
    def __init__(self, limit = 12, min_change = 0):
        self.limit = limit
        self.min_change = min_change
        self.min_loss = np.inf
        self.counter = 0

    def __call__(self, validation_loss):
        if validation_loss < self.min_loss:
            self.min_loss = validation_loss
            self.counter = 0
        elif validation_loss > self.min_loss + self.min_change:
            self.counter += 1
            if self.counter >= self.limit:
                return True
        return False

def save_accuracy_plot(accuracies, n_samples, method, figs_path):
    # Save plot of accuracy vs number of samples in training dataset
    plt.cla()
    y_min = np.min(accuracies) - 0.1
    y_max = np.max(accuracies) + 0.1
    xi = list(range(len(n_samples)))
    plt.ylim(y_min, y_max)
    plt.plot(xi, accuracies, marker="o", linestyle="--", color="b")
    plt.xlabel("Number of samples trained on")
    plt.ylabel("Test accuracy")
    plt.xticks(xi, n_samples, rotation=90)
    plt.title(method)
    plt.grid()
    plt.tight_layout()
    filename = "accuracy_" + method + ".png"
    filename = join(figs_path, filename)
    plt.savefig(filename)
    print(f"Saved plot to {filename}.")

class RandomAugmentationModule:
    """
    For generating random transforms for SimCLR training
    """
    def __init__(self, image_size = (128, 128)):
        self.image_size = image_size
    
    def generate_transform(self):
        tfs = []
        tfs.append(transforms.RandomRotation(180, fill=1))
        tfs.append(transforms.RandomResizedCrop(self.image_size))
        tfs.append(transforms.ColorJitter(brightness=.2, hue=.3))
        gaussian_kernel_size = 2 * random.randint(0,5) + 1
        tfs.append(transforms.GaussianBlur(gaussian_kernel_size))
        return transforms.Compose(tfs)
