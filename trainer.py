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
from sklearn.metrics import balanced_accuracy_score, top_k_accuracy_score
from sklearn.neural_network import MLPClassifier

# Lots of repeating code here...
# TODO: Create a universal BaseTrainer class
# Split into train step, validation step, etc.
# Train a simple linear classifier on non-classifier-based methods to evaluate performance
# For classifier-based methods (softmax and arcface), just use the class. head to evaluate.
# NOTE: Switch to linear-classifier-on-embeddings for validation accuracy for ALL models!
# Also, train_history should be the same (but can have empty lists for some keys)
# Let the plot function determine how to plot things depending on what it gets

class BaseTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, loss_function, optimizer, max_epochs, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.device = device
        self.val_steps = len(self.train_dataloader) // 5
        self.early_stopper = EarlyStopper()
        self.train_history = {"train_loss":[], "train_accuracy":[], "val_loss":[], "val_accuracy":[]}

    def train(self):
        pass

    def train_step(self):
        pass

    def compute_loss(self):
        pass

    def validation_step(self):
        pass

    def validate(self):
        pass

    def save_plot(self, filename):
        pass

def train_classifier(model, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device):
    train_history = {"train_loss":[], "train_accuracy":[], "val_loss":[], "val_accuracy":[]}
    steps = len(train_dataloader) // 5 #Compute validation and train loss 5 times every epoch
    earlystop = EarlyStopper()
    for epoch in range(epochs):
        train_losses = []
        train_accuracies = []
        for i, data in enumerate((pbar := tqdm(train_dataloader))):
            images, labels  = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            train_accuracies.append((predicted == labels).sum().item() / labels.size(0))
            if i % steps == steps - 1:
                train_loss = np.mean(train_losses)
                train_losses = []
                train_accuracy = 100.0 * np.mean(train_accuracies)
                train_accuracies = []
                correct = 0
                total = 0
                val_losses = []
                model.eval()
                with torch.no_grad():
                    for data in val_dataloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = model(images)
                        val_losses.append(loss_function(outputs, labels).item())
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    val_accuracy = 100.0 * correct / total
                    val_loss = np.mean(val_losses)
                if train_history["val_accuracy"] and val_accuracy > np.max(train_history["val_accuracy"]):
                    # Save weights for backbone only (not head)
                    torch.save(model[0].state_dict(), join(checkpoints_path, "best.pth"))
                train_history["train_loss"].append(train_loss)
                train_history["train_accuracy"].append(train_accuracy)
                train_history["val_loss"].append(val_loss)
                train_history["val_accuracy"].append(val_accuracy)
                pbar_string = f"Epoch {epoch}/{epochs-1} | Loss: Train={train_loss:.3f} Val={val_loss:.3f} | Acc.: Train={train_accuracy:.1f}% Val={val_accuracy:.1f}%"
                pbar.set_description(pbar_string)
                if earlystop(val_loss):
                    print(f"Early stopped at epoch {epoch}")
                    return train_history
                model.train()
    return train_history


def train_triplet(model, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device):
    train_history = {"train_loss":[], "val_loss":[]}
    steps = len(train_dataloader) // 5 #Compute validation and train loss 5 times every epoch
    negative_policy = "semi-hard"
    positive_policy = "easy"
    for epoch in range(epochs):
        train_losses = []
        val_loss = 0
        train_loss = 0

        if epoch >= 10: # Increase difficulty after some epochs to prevent collapse
            negative_policy = "hard"
        if epoch >= 25:
            positive_policy = "hard"
        
        for i, data in enumerate((pbar := tqdm(train_dataloader))):
            images, labels  = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_function(outputs, labels, negative_policy, positive_policy)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            if i % steps == steps - 1:
                train_losses = []
                val_losses = []
                model.eval()
                with torch.no_grad():
                    for data in val_dataloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        outputs = model(images)
                        val_losses.append(loss_function(outputs, labels, negative_policy="hard", positive_policy="hard").item())
                    val_loss = np.mean(val_losses)
                loss_function.mine_hard_triplets = False
                train_history["train_loss"].append(train_loss)
                train_history["val_loss"].append(val_loss)
                model.train()
            if train_losses:
                train_loss = np.mean(train_losses)
            pbar_string = f"Epoch {epoch}/{epochs-1} | TripletLoss: Train={train_loss:.3f} Val={val_loss:.3f}"
            pbar.set_description(pbar_string)
        torch.save(model[0].state_dict(), join(checkpoints_path, "best.pth"))
    return train_history

def train_arcface(model, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device):
    train_history = {"train_loss":[], "train_accuracy":[], "val_loss":[], "val_accuracy":[]}
    steps = len(train_dataloader) // 5 #Compute validation and train loss 5 times every epoch
    earlystop = EarlyStopper()
    for epoch in range(epochs):
        train_losses = []
        train_accuracies = []
        for i, data in enumerate((pbar := tqdm(train_dataloader))):
            images, labels  = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            embeddings = model[0](images)
            outputs = model[1](embeddings)
            weights = model[1].get_weights(normalize=True)
            loss = loss_function(embeddings, weights, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            _, predicted = torch.max(outputs.data, 1)
            train_accuracies.append((predicted == labels).sum().item() / labels.size(0))
            if i % steps == steps - 1:
                train_loss = np.mean(train_losses)
                train_losses = []
                train_accuracy = 100.0 * np.mean(train_accuracies)
                train_accuracies = []
                correct = 0
                total = 0
                val_losses = []
                model.eval()
                with torch.no_grad():
                    for data in val_dataloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        embeddings = model[0](images)
                        outputs = model[1](embeddings)
                        weights = model[1].get_weights(normalize=True)
                        val_losses.append(loss_function(embeddings, weights, labels).item())
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                    val_accuracy = 100.0 * correct / total
                    val_loss = np.mean(val_losses)
                if train_history["val_accuracy"] and val_accuracy > np.max(train_history["val_accuracy"]):
                    torch.save(model[0].state_dict(), join(checkpoints_path, "best.pth"))
                train_history["train_loss"].append(train_loss)
                train_history["train_accuracy"].append(train_accuracy)
                train_history["val_loss"].append(val_loss)
                train_history["val_accuracy"].append(val_accuracy)
                pbar_string = f"Epoch {epoch}/{epochs-1} | Loss: Train={train_loss:.3f} Val={val_loss:.3f} | Acc.: Train={train_accuracy:.1f}% Val={val_accuracy:.1f}%"
                pbar.set_description(pbar_string)
                if earlystop(val_loss):
                    print(f"Early stopped at epoch {epoch}")
                    return train_history
                model.train()
    return train_history

def train_simclr(model, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device):
    train_history = {"train_loss":[], "val_loss":[]}
    steps = len(train_dataloader) // 5 #Compute validation and train loss 5 times every epoch
    RAM = RandomAugmentationModule()
    for epoch in range(epochs):
        train_losses = []
        val_loss = 0
        train_loss = 0
        
        for i, data in enumerate((pbar := tqdm(train_dataloader))):
            images, labels  = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            
            #SimCLR forward
            t1 = RAM.generate_transform()
            t2 = RAM.generate_transform()
            v1 = t1(images)
            v2 = t2(images)
            z1 = model(v1)
            z2 = model(v2)
            loss = loss_function(z1, z2)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
            if i % steps == steps - 1:
                train_losses = []
                val_losses = []
                model.eval()
                cos_embedding_loss = nn.CosineEmbeddingLoss() # We use this for validation loss
                with torch.no_grad():
                    for data in val_dataloader:
                        images, labels = data[0].to(device), data[1].to(device)
                        t1 = RAM.generate_transform()
                        t2 = RAM.generate_transform()
                        v1 = t1(images)
                        v2 = t2(images)
                        z1 = model(v1)
                        z2 = model(v2)
                        val_losses.append(loss_function(z1, z2).item())
                    val_loss = np.mean(val_losses)
                train_history["train_loss"].append(train_loss)
                train_history["val_loss"].append(val_loss)
                model.train()
            if train_losses:
                train_loss = np.mean(train_losses)
            pbar_string = f"Epoch {epoch}/{epochs-1} | NTXentLoss: Train={train_loss:.3f} Val={val_loss:.3f}"
            pbar.set_description(pbar_string)
        torch.save(model[0].state_dict(), join(checkpoints_path, "best.pth"))
    return train_history
