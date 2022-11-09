import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models import efficientnet_v2_s
from torchvision.models import EfficientNet_V2_S_Weights

def init_weights(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

class ConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
                    nn.Conv2d(
                        in_channels=64,
                        out_channels=64,
                        kernel_size=3,
                        stride=1,
                        padding=1
                        ),
                    nn.BatchNorm2d(64),
                    nn.ReLU()
        )
        self.layers.apply(init_weights)

    def forward(self, x):
        return self.layers(x)

class ClassifierHead(nn.Module):
    """
        For use with SoftMax classifier and ArcFace (or other classifier-based methods)
    """
    def __init__(self, embedding_dimension, number_of_classes):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_classes = number_of_classes
        self.layers = nn.Sequential(nn.Linear(embedding_dimension, self.number_of_classes))
        self.layers.apply(init_weights)

    def forward(self, x):
        x = self.layers(x)
        return x

    def get_weights(self, normalize=True):
        # Used to compute Angular Margin Loss when training ArcFace
        weights = self.layers[0].weight
        if normalize:
            return nn.functional.normalize(weights, p=2, dim=1) #L2-normalize weights
        return weights 

class ProjectionHead(nn.Module):
    """
        For use with SimCLR
    """
    def __init__(self, embedding_dimension, hidden_dimension=None, output_dimension=None):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        if output_dimension:
            self.output_dimension = output_dimension
        else:
            self.output_dimension = self.embedding_dimension
        if hidden_dimension:
            self.hidden_dimension = hidden_dimension
        else:
            self.hidden_dimension = self.embedding_dimension
        self.layers = nn.Sequential(
                        nn.Linear(self.embedding_dimension, self.hidden_dimension),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dimension, self.hidden_dimension),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dimension, self.output_dimension)
        )
        self.layers.apply(init_weights)

    def forward(self, x):
        x = self.layers(x)
        return x

class BackBone(nn.Module):
    def __init__(self, embedding_dimension=128):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        #Use EfficientNet V2 with small weights as base
        pretrained = efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        #Cut off last layers
        self.feature_extractor = nn.Sequential(*list(list(pretrained.children())[:-2][0].children())[:-4])
        for param in self.feature_extractor.parameters():
                param.requires_grad = True #False
        #Extra conv layers. Output (-1, 64, 8, 8)
        self.extra_layers = nn.Sequential(
                    ConvBlock(),
                    ConvBlock(),
                    nn.MaxPool2d(kernel_size=2, stride=2))
        #Fully connected layers.
        self.fc1 = nn.Sequential(
                nn.Linear(4096, self.embedding_dimension), 
                nn.LeakyReLU(),
                nn.Dropout(0.2))
        self.fc1.apply(init_weights)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.extra_layers(x)
        x = torch.flatten(x, start_dim = 1)
        emb = self.fc1(x)
        emb = nn.functional.normalize(emb, p=2, dim=1) #L2-normalize embeddings
        return emb
