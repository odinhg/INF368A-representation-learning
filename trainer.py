import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from utilities import EarlyStopper, RandomAugmentationModule
from os.path import join
from configfile import *
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import top_k_accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt

class BaseTrainer:
    def __init__(self):
        pass

    def init(self, model, train_dataloader, val_dataloader, loss_function, optimizer, max_epochs, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.device = device
        self.val_steps = len(self.train_dataloader) // 5
        self.early_stopper = EarlyStopper()
        self.train_history = {"train_loss":[], "val_accuracy_top1":[], "val_accuracy_top3":[]}
        self.current_epoch = 0

    def train(self, checkpoint_filename):
        for epoch in range(self.max_epochs):
            self.current_epoch = epoch
            train_losses = []
            for i, data in enumerate((pbar := tqdm(self.train_dataloader))):
                # Train network
                train_loss = self.train_step(data)
                train_losses.append(train_loss)

                if i % self.val_steps == self.val_steps - 1: # Time to validate
                    mean_train_loss = np.mean(train_losses)
                    self.train_history["train_loss"].append(mean_train_loss)
                    train_losses = []
                    self.model.eval()
                    val_accuracy = self.validate()
                    if val_accuracy >= np.max(self.train_history["val_accuracy_top1"]):
                        torch.save(self.model[0].state_dict(), checkpoint_filename)
                    if self.early_stopper(val_accuracy):
                        print(f"Early stopped at epoch {epoch}")
                        return
                    pbar_str = f"Epoch {epoch:02}/{self.max_epochs:02} | Train Loss = {mean_train_loss:.4f} | Val Acc. = {100*val_accuracy:.2f}% | ES {self.early_stopper.counter:02}/{self.early_stopper.limit:02}"
                    pbar.set_description(pbar_str)
                    self.model.train()
                    
    def train_step(self, data):
        images, labels = data[0].to(self.device), data[1].to(self.device)
        self.optimizer.zero_grad()
        loss = self.compute_loss(images, labels)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def compute_loss(self, images, labels):
        # To be implemented for each architecture
        pass

    def validate(self):
        # Compute embeddings for validation data
        with torch.no_grad():
            embeddings = []
            labels = []
            for data in self.val_dataloader:
                images, batch_labels = data[0].to(self.device), data[1].to(self.device)
                batch_embeddings = self.model[0](images)
                embeddings += batch_embeddings.cpu().detach().tolist()
                labels += batch_labels.cpu().detach().tolist()

        # Split validation data into train and test
        embeddings = pd.DataFrame(embeddings)
        labels = pd.Series(labels)
        X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.35, random_state=420)

        # Train validation classifier on embeddings and compute accuracies
        val_classifier = make_pipeline(StandardScaler(), SVC(gamma="auto", probability=True))
        val_classifier.fit(X_train, y_train)
        y_pred_proba = val_classifier.predict_proba(X_test)
        accuracy_top1 = top_k_accuracy_score(y_test, y_pred_proba, k=1)
        accuracy_top3 = top_k_accuracy_score(y_test, y_pred_proba, k=3)
        self.train_history["val_accuracy_top1"].append(accuracy_top1)
        self.train_history["val_accuracy_top3"].append(accuracy_top3)
        
        return accuracy_top1

    def save_plot(self, filename):
        # Plot losses and accuracies from training
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
        axes[0].plot(self.train_history["train_loss"], 'b', label="Train Loss")
        axes[1].plot(self.train_history["val_accuracy_top1"], 'b', label="Top-1 Validation Accuracy")
        axes[1].plot(self.train_history["val_accuracy_top3"], 'g', label="Top-3 Validation Accuracy")
        axes[0].title.set_text('Loss')
        axes[1].title.set_text('Accuracy')
        axes[0].legend(loc="upper right")
        axes[1].legend(loc="upper right")
        fig.tight_layout()
        plt.savefig(filename)

class SimCLRTrainer(BaseTrainer):
    """
        Trainer class for SimCLR self-supervised learning
    """
    def compute_loss(self, images, labels):
        RAM = RandomAugmentationModule()
        t1 = RAM.generate_transform()
        t2 = RAM.generate_transform()
        v1 = t1(images)
        v2 = t2(images)
        z1 = self.model(v1)
        z2 = self.model(v2)
        return self.loss_function(z1, z2)

class XFaceTrainer(BaseTrainer):
    """
        Trainer class for ArcFace, CosFace and SphereFace
    """
    def compute_loss(self, images, labels):
        embeddings = self.model[0](images)
        outputs = self.model[1](embeddings)
        weights = self.model[1].get_weights()
        return self.loss_function(embeddings, weights, labels)

class ClassifierTrainer(BaseTrainer):
    """
        Trainer class for Softmax classifier
    """
    def compute_loss(self, images, labels):
        outputs = self.model(images)
        return self.loss_function(outputs, labels)

class TripletTrainer(BaseTrainer):
    """
        Trainer class for Triplet Margin Loss
    """
    def init(self, *args, **kwargs):
        super(TripletTrainer, self).init(*args, **kwargs)
        # Start training with easy positives and semi-hard negatives
        self.positive_policy = "easy"
        self.negative_policy = "semi-hard"
        self.hard_negatives_epoch = 10 # At which epoch to switch to hard negatives
        self.hard_positives_epoch = 25 # At which epoch to switch to hard positives

    def set_mining_policy(self, epoch):
        if epoch >= self.hard_negatives_epoch:
            self.negative_policy = "hard"
        if epoch >= self.hard_positives_epoch:
            self.positive_policy = "hard"

    def compute_loss(self, images, labels):
        # Set mining policy based on epoch and compute loss
        self.set_mining_policy(self.current_epoch)
        outputs = self.model(images)
        return self.loss_function(outputs, labels, self.negative_policy, self.positive_policy)
