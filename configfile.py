import torch
import torch.nn as nn
import torch.optim as optim
from os import listdir, makedirs
from os.path import isfile, join, exists
from dataloader import FlowCamDataLoader
from backbone import BackBone, ClassifierHead, ProjectionHead
from loss_functions import TripletLoss, AngularMarginLoss, NTXentLoss
from trainer import SimCLRTrainer, ClassifierTrainer, XFaceTrainer, TripletTrainer

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
class_names_all = ['Eutintinnus', 'Rhabdonellidae', 'Rhizosoleniaceae', 'Undellidae', 'Odontella', 'Thalassionematales', 'Planktoniella sol', 'Dinophyceae', 'pennate', 'badfocus', 'other<living', 'chainthin', 'Steenstrupiella', 'Bacillariophyceae', 'darkrods', 'Codonellopsis', 'Neoceratium pentagonum', 'cyano a', 'Rhizosolenia inter. Richelia', 'Nassellaria', 'Neoceratium fusus', 'part<Odontella', 'Ditylum', 'part<Crustacea', 'lightrods', 'darksphere', 'Bacteriastrum', 'nauplii', 'Richelia', 'Climacodium', 'Retaria', 'ball_bearing_like', 'Neoceratium', 'chainlarge', 'Melosiraceae', 'contrasted_blob', 'artefact', 'part<Ditylum', 'crumple sphere', 'Protoperidinium', 'Coscinodiscaceae', 'Tintinnina', 'UCYNA like', 'Chaetoceros']

number_of_classes_total = len(class_names_all)
number_of_unseen_classes = 10 # Set the number of classes we exclude from training
class_idx = list(range(0, number_of_classes_total - number_of_unseen_classes))
class_idx_unseen = list(range(number_of_classes_total - number_of_unseen_classes, number_of_classes_total))
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
    trainer = TripletTrainer()
elif model_type == "arcface":
    # Angular Margin Loss Model
    loss_function = AngularMarginLoss(m=margin, s=scale, number_of_classes=number_of_classes)
    head = ClassifierHead(embedding_dimension, number_of_classes)
    trainer = XFaceTrainer()
elif model_type == "simclr":
    # SimCLR based Model
    loss_function = NTXentLoss(t=temperature)
    head = ProjectionHead(embedding_dimension)
    trainer = SimCLRTrainer()
else:
    # Classic SoftMax Classifier Model
    loss_function = nn.CrossEntropyLoss()
    head = ClassifierHead(embedding_dimension, number_of_classes)
    trainer = ClassifierTrainer()

if head:
    model = nn.Sequential(backbone, head)
else:
    model = nn.Sequential(backbone)

optimizer = optim.Adam(model.parameters(), lr=lr)

device = torch.device('cuda:4') 

trainer.init(model, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device)

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
