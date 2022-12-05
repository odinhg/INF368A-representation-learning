import torch
import numpy as np
import pandas as pd
from os.path import isfile, join
from tqdm import tqdm
from configfile import *
from utilities import save_train_plot, save_loss_plot
from dataloader import FlowCamDataLoader
from trainer import SimCLRTrainer, ClassifierTrainer, XFaceTrainer, TripletTrainer
from torchsummary import summary

if __name__ == "__main__":
    summary(model, (3, *image_size), device=device)
    model.to(device)
    print(f"Training model: {model_type}")
    trainer.train(checkpoint_filename=join(checkpoints_path, "best.pth"))
    trainer.save_plot(join(figs_path, "training_plot.png"))
