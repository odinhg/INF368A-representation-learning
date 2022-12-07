import torch
import torch.nn as nn
import torch.optim as optim
from os import listdir, makedirs
from os.path import isfile, join, exists
from dataloader import FlowCamDataLoader, FlowCamDataSet
from backbone import BackBone, ClassifierHead, ProjectionHead
from loss_functions import TripletLoss, AngularMarginLoss, NTXentLoss, LargeMarginCosineLoss
from trainer import SimCLRTrainer, ClassifierTrainer, XFaceTrainer, TripletTrainer

torch.manual_seed(0)

# Global settings (for all models)
image_size = (128, 128)
embedding_dimension = 128
backbone = BackBone(embedding_dimension)
num_workers = 8

# Dataset selection 
class_names_all = ["artefact","Bacillariophyceae","cyano a","Chaetoceros","dark","light","Melosiraceae","nauplii","Neoceratium pentagonum", "detritus", "contrasted_blob", "Dinophyceae", "Coscinodiscaceae", "part<Crustacea", "fiber", "lightrods", "lightsphere", "darksphere", "cyano b", "chainthin"]
number_of_classes_total = len(class_names_all)
number_of_unseen_classes = 10 # Set the number of classes we exclude from training data
class_idx = list(range(0, number_of_classes_total - number_of_unseen_classes))
class_idx_unseen = list(range(number_of_classes_total - number_of_unseen_classes, number_of_classes_total))
class_names = [class_names_all[i] for i in class_idx]
class_names_unseen = [class_names_all[i] for i in class_idx_unseen]
number_of_classes = len(class_names)

# Small dataset for training validation support vector classifier
validation_classes = ["Hemiaulus", "Gymnodiniales", "Copepoda", "Dictyocysta", "Spumellaria", "Foraminifera", "Ornithocercus"]

# Model specific configurations
models = {}

# SimCLR Model
models["SimCLR"] = {
        "head" : ProjectionHead(embedding_dimension),
        "loss_function" : NTXentLoss(t=0.2),
        "trainer" : SimCLRTrainer(),
        "batch_size" : 1024,
        "epochs" : 20,
        "lr" : 0.0015,
        "device" : "cuda:3",
        "balance_train_data" : False
        }

# SimCLR Model (With balanced dataloader)
models["BalancedSimCLR"] = {
        "head" : ProjectionHead(embedding_dimension),
        "loss_function" : NTXentLoss(t=0.2),
        "trainer" : SimCLRTrainer(),
        "batch_size" : 1024,
        "epochs" : 20,
        "lr" : 0.0015,
        "device" : "cuda:3",
        "balance_train_data" : False
        }

# Triplet Margin Loss Model
models["TripletMarginLoss"] = {
        "head" : None,
        "loss_function" : TripletLoss(margin=0.2),
        "trainer" : TripletTrainer(),
        "batch_size" : 128,
        "epochs" : 20,
        "lr" : 0.0015,
        "device" : "cuda:4",
        "balance_train_data" : False
        }

# ArcFace Model
models["ArcFace"] = {
        "head" : ClassifierHead(embedding_dimension, number_of_classes),
        "loss_function" : AngularMarginLoss(m=0.1, s=64, number_of_classes=number_of_classes),
        "trainer" : XFaceTrainer(),
        "batch_size" : 128,
        "epochs" : 20,
        "lr" : 0.0015,
        "device" : "cuda:5",
        "balance_train_data" : False
        }

# CosFace Model
models["CosFace"] = {
        "head" : ClassifierHead(embedding_dimension, number_of_classes),
        "loss_function" : LargeMarginCosineLoss(m=0.1, s=64, number_of_classes=number_of_classes),
        "trainer" : XFaceTrainer(),
        "batch_size" : 128,
        "epochs" : 20,
        "lr" : 0.0015,
        "device" : "cuda:6",
        "balance_train_data" : False
        }

# Standard Softmax Classifier Model
models["Softmax"] = {
        "head" : ClassifierHead(embedding_dimension, number_of_classes),
        "loss_function" : nn.CrossEntropyLoss(),
        "trainer" : ClassifierTrainer(),
        "batch_size" : 128,
        "epochs" : 20,
        "lr" : 0.0015,
        "device" : "cuda:7",
        "balance_train_data" : False
        }

# Select and load model
model_names = list(models.keys())
for i,k in enumerate(model_names):
    print(f"[{i}] {k}")
choice = int(input("Choose model: "))
config_name = model_names[choice]
selected_model = models[config_name]

device = torch.device(selected_model["device"]) 
head = selected_model["head"]
loss_function = selected_model["loss_function"]
trainer = selected_model["trainer"]
batch_size = selected_model["batch_size"]
epochs = selected_model["epochs"]
lr = selected_model["lr"]
balance_train_data = selected_model["balance_train_data"]

#Load custom dataset
#data = FlowCamDataLoader(class_names, image_size, val, test,  batch_size)
#train_dataloader = FlowCamDataLoader(class_names, image_size, val, test,  batch_size, split=False)
#train_dataloader = data["train_dataloader"]
#val_dataloader = data["val_dataloader"]
#test_dataloader = data["test_dataloader"]
#train_dataset = data["train_dataset"]
#val_dataset = data["val_dataset"]
#test_dataset = data["test_dataset"]
train_dataset = FlowCamDataSet(class_names, image_size)
if balance_train_data:
    sample_weights = train_dataset.get_sample_weights()
    sampler = torch.utils.data.WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
else:
    sampler = None
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, sampler=sampler)

val_dataset = FlowCamDataSet(validation_classes, image_size)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
