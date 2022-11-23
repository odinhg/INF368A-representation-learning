import torch
import numpy as np
import pandas as pd
from os.path import isfile, join
from tqdm import tqdm
from configfile import *
from utilities import save_train_plot, save_loss_plot
from dataloader import FlowCamDataLoader
from trainer import train_classifier, train_triplet, train_arcface, train_simclr
from trainer import SimCLRTrainer, ClassifierTrainer
from torchsummary import summary

if __name__ == "__main__":
    summary(model, (3, *image_size), device=device)
    model.to(device)

    print(f"Training model: {model_type}")
    if model_type == "triplet":
        train_history = train_triplet(model, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device)
        save_loss_plot(join(figs_path, "training_plot.png"), train_history)
    elif model_type == "arcface":
        train_history = train_arcface(model, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device)
        save_train_plot(join(figs_path, "training_plot.png"), train_history)
    elif model_type == "simclr":
        trainer = SimCLRTrainer(model, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device)
        trainer.train(checkpoint_filename=join(checkpoints_path, "best.pth"))
        trainer.save_plot(join(figs_path, "training_plot.png"))
    else:
        trainer = ClassifierTrainer(model, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device)
        trainer.train(checkpoint_filename=join(checkpoints_path, "best.pth"))
        trainer.save_plot(join(figs_path, "training_plot.png"))
        #train_history = train_classifier(model, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device)
        #save_train_plot(join(figs_path, "training_plot.png"), train_history)
