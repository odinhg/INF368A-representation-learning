import torch
import torch.nn as nn
import torch.optim as optim
from os import listdir, makedirs
from os.path import isfile, join, exists
from dataloader import FlowCamDataLoader
from backbone import BackBone, ClassifierHead, ProjectionHead
from loss_functions import TripletLoss, AngularMarginLoss, NTXentLoss, LargeMarginCosineLoss
from trainer import SimCLRTrainer, ClassifierTrainer, XFaceTrainer, TripletTrainer

torch.manual_seed(0)

# Global settings (for all models)
val = 0.05
test = 0.00 
image_size = (128, 128)
embedding_dimension = 128
backbone = BackBone(embedding_dimension)
device = torch.device('cuda:4') 

# Dataset selection 
class_names_all = ["artefact","Bacillariophyceae","cyano a","Chaetoceros","dark","light","Melosiraceae","nauplii","Neoceratium pentagonum", "detritus", "contrasted_blob", "Dinophyceae", "Coscinodiscaceae", "part<Crustacea", "fiber", "lightrods", "lightsphere", "darksphere", "cyano b", "chainthin"]
number_of_classes_total = len(class_names_all)
number_of_unseen_classes = 10 # Set the number of classes we exclude from training data
class_idx = list(range(0, number_of_classes_total - number_of_unseen_classes))
class_idx_unseen = list(range(number_of_classes_total - number_of_unseen_classes, number_of_classes_total))
class_names = [class_names_all[i] for i in class_idx]
class_names_unseen = [class_names_all[i] for i in class_idx_unseen]
number_of_classes = len(class_names)

# Model specific configurations
models = {}

# SimCLR Model
models["SimCLR"] = {
        "head" : ProjectionHead(embedding_dimension),
        "loss_function" : NTXentLoss(t=0.1),
        "trainer" : SimCLRTrainer(),
        "batch_size" : 256,
        "epochs" : 50,
        "lr" : 0.0015
        }

# Triplet Margin Loss Model
models["TripletMarginLoss"] = {
        "head" : None,
        "loss_function" : TripletLoss(margin=0.5),
        "trainer" : TripletTrainer(),
        "batch_size" : 128,
        "epochs" : 50,
        "lr" : 0.0015
        }

# ArcFace Model
models["ArcFace"] = {
        "head" : ClassifierHead(embedding_dimension, number_of_classes),
        "loss_function" : AngularMarginLoss(m=0.5, s=64, number_of_classes=number_of_classes),
        "trainer" : XFaceTrainer(),
        "batch_size" : 128,
        "epochs" : 50,
        "lr" : 0.0015
        }

# CosFace Model
models["CosFace"] = {
        "head" : ClassifierHead(embedding_dimension, number_of_classes),
        "loss_function" : LargeMarginCosineLoss(m=0.5, s=64, number_of_classes=number_of_classes),
        "trainer" : XFaceTrainer(),
        "batch_size" : 128,
        "epochs" : 50,
        "lr" : 0.0015
        }

# Standard Softmax Classifier Model
models["Softmax"] = {
        "head" : ClassifierHead(embedding_dimension, number_of_classes),
        "loss_function" : nn.CrossEntropyLoss(),
        "trainer" : ClassifierTrainer(),
        "batch_size" : 128,
        "epochs" : 50,
        "lr" : 0.0015
        }

# Select and load model
model_names = list(models.keys())
for i,k in enumerate(model_names):
    print(f"[{i}] {k}")
choice = int(input("Choose model: "))
config_name = model_names[choice]
selected_model = models[config_name]

head = selected_model["head"]
loss_function = selected_model["loss_function"]
trainer = selected_model["trainer"]
batch_size = selected_model["batch_size"]
epochs = selected_model["epochs"]
lr = selected_model["lr"]

#Load custom dataset
data = FlowCamDataLoader(class_names, image_size, val, test,  batch_size)
train_dataloader = data["train_dataloader"]
val_dataloader = data["val_dataloader"]
test_dataloader = data["test_dataloader"]
train_dataset = data["train_dataset"]
val_dataset = data["val_dataset"]
test_dataset = data["test_dataset"]
    
# Assemble backbone and head
if head:
    model = nn.Sequential(backbone, head)
else:
    model = nn.Sequential(backbone)

optimizer = optim.Adam(model.parameters(), lr=lr)
trainer.init(model, train_dataloader, val_dataloader, loss_function, optimizer, epochs, device)

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
