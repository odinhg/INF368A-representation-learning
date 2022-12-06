import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn as nn
from os.path import isfile, join
from tqdm import tqdm
from configfile import *
from utilities import save_embeddings
from dataloader import FlowCamDataLoader

if __name__ == "__main__":
    if not isfile(join(checkpoints_path, "best.pth")):
        exit("No checkpoint found! Please run training before evaluating model.")
    model[0].to(device)
    #Load the classes that the model has never seen before
    unseen_dataloader = FlowCamDataLoader(class_names_unseen, image_size=image_size, batch_size=batch_size, split=False)
    print("Loading checkpoint.")
    backbone.load_state_dict(torch.load(join(checkpoints_path, "best.pth")))
    
    print("Embedding data.")
    save_embeddings(backbone, class_idx, train_dataloader, embeddings_file_train, device)
    #save_embeddings(backbone, class_idx, test_dataloader, embeddings_file_test, device)
    save_embeddings(backbone, class_idx_unseen, unseen_dataloader, embeddings_file_unseen, device)
