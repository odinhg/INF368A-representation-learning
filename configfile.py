import torch
import torch.nn as nn
import torch.optim as optim
from os import listdir, makedirs
from os.path import isfile, join, exists
from dataloader import FlowCamDataLoader
from backbone import BackBone, ClassifierHead, ProjectionHead
from loss_functions import TripletLoss, AngularMarginLoss, NTXentLoss

torch.manual_seed(0)

configfiles = [f.split(".")[0] for f in listdir("configs") if isfile(join("configs", f)) and f[0].isalpha()]

print("Select configuration file to load:")
idx = -1
while not (0 <= idx < len(configfiles)):
    for i, f in enumerate(configfiles):
        print(f"[{i}]\t{f}")    
    idx = int(input("Config: "))

exec(f"from configs.{configfiles[idx]} import *") #Ugly, but it works for now.

# Classes to use for training and unseen classes
class_names_all = ["chainthin", "darksphere", "Rhabdonellidae", "Odontella", "Codonellopsis", "Neoceratium", "Retaria", "Thalassionematales", "Chaetoceros"]
class_idx = [0, 1, 2, 3, 4, 5] 
class_idx_unseen = [6, 7, 8]
class_names = [class_names_all[i] for i in class_idx]
class_names_unseen = [class_names_all[i] for i in class_idx_unseen]
number_of_classes = len(class_names)

#Load custom dataset
data = FlowCamDataLoader(class_names, image_size, val, test,  batch_size)
train_dataloader = data["train_dataloader"]
val_dataloader = data["val_dataloader"]
test_dataloader = data["test_dataloader"]
train_dataset = data["train_dataset"]
val_dataset = data["val_dataset"]
test_dataset = data["test_dataset"]
    
# Model
embedding_dimension = 128
backbone = BackBone(embedding_dimension)

if model_type == "triplet":
    # Triplet Margin Loss Model
    loss_function = TripletLoss(margin=margin)
    head = None
elif model_type == "arcface":
    # Angular Margin Loss Model
    loss_function = AngularMarginLoss(m=margin, s=scale, number_of_classes=number_of_classes)
    head = ClassifierHead(embedding_dimension, number_of_classes) 
elif model_type == "simclr":
    # SimCLR based Model
    loss_function = NTXentLoss(t=temperature)
    head = ProjectionHead(embedding_dimension)
else:
    # Classic SoftMax Classifier Model
    loss_function = nn.CrossEntropyLoss()
    head = ClassifierHead(embedding_dimension, number_of_classes) 

if head:
    model = nn.Sequential(backbone, head)
else:
    model = nn.Sequential(backbone)

optimizer = optim.Adam(model.parameters(), lr=lr)

device = torch.device('cuda:4') 

config_name = configfiles[idx]
embeddings_path = join("embeddings", config_name)
embeddings_file_train = join(embeddings_path, "embeddings_train.pkl")
embeddings_file_test = join(embeddings_path, "embeddings_test.pkl")
embeddings_file_unseen = join(embeddings_path, "embeddings_unseen.pkl")
figs_path = join("figs", config_name)
checkpoints_path = join("checkpoints", config_name)

if not exists(embeddings_path):
    makedirs(embeddings_path)
if not exists(figs_path):
    makedirs(figs_path)
if not exists(checkpoints_path):
    makedirs(checkpoints_path)
