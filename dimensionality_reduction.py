import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import isfile, join
import umap.umap_ as umap
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont
from configfile import * 
from dataloader import FlowCamDataSet
from utilities import sample_df

if __name__ == "__main__":
    if not (isfile(embeddings_file_train) and isfile(embeddings_file_test) and isfile(embeddings_file_unseen)):
        exit("Embeddings not found. Please evaluate model first!")
   
    # Load embeddings
    df_test = pd.read_pickle(embeddings_file_test)
    df_train = pd.read_pickle(embeddings_file_train)
    df_unseen = pd.read_pickle(embeddings_file_unseen)
    
    # Randomly subsample images
    number_of_samples = 2500
    df_test = sample_df(df_test, number_of_samples)
    df_train = sample_df(df_train, number_of_samples)
    df_unseen = sample_df(df_unseen, number_of_samples)
    df_all = pd.concat([df_test, df_train, df_unseen])
    
    # Standardize features
    standard_scaler = StandardScaler().fit(df_all.iloc[:,2:])
    df_test.iloc[:,2:] = standard_scaler.transform(df_test.iloc[:,2:])
    df_train.iloc[:,2:] = standard_scaler.transform(df_train.iloc[:,2:])
    df_unseen.iloc[:,2:] = standard_scaler.transform(df_unseen.iloc[:,2:])
    df_all.iloc[:,2:] = standard_scaler.transform(df_all.iloc[:,2:])

    # Fit UMAP and reduce dimensions
    reducer = umap.UMAP(verbose=True)
    reducer.fit(df_all.iloc[:,2:])
    df_projection_test = reducer.transform(df_test.iloc[:,2:])
    df_projection_train = reducer.transform(df_train.iloc[:,2:])
    df_projection_unseen = reducer.transform(df_unseen.iloc[:,2:])

    # Generate and save UMAP plots
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    s = ax[0].scatter(df_projection_test[:,0], df_projection_test[:,1], c=df_test.label_idx, s=5)
    ax[0].set_aspect("equal", "datalim")
    ax[0].set_title("UMAP (embedded test data)")
    ax[0].legend(handles=s.legend_elements()[0], labels=class_names)
    s = ax[1].scatter(df_projection_train[:,0], df_projection_train[:,1], c=df_train.label_idx, s=5)
    ax[1].set_aspect("equal", "datalim")
    ax[1].set_title("UMAP (embedded train data)")
    ax[1].legend(handles=s.legend_elements()[0], labels=class_names)
    s = ax[2].scatter(df_projection_unseen[:,0], df_projection_unseen[:,1], c=df_unseen.label_idx, s=5)
    ax[2].set_aspect("equal", "datalim")
    ax[2].set_title("UMAP (embedded unseen classes)")
    ax[2].legend(handles=s.legend_elements()[0], labels=class_names_unseen)
    fig.tight_layout()
    plt.savefig(join(figs_path, "umap_embeddings.png"))
    
    # The following code is a bit messy, and will probably never get cleaned up. So sorry about that.
    # Find samples closest to and furthest away from class center
    dataset = FlowCamDataSet(class_names, image_size)
    df_projections_train = pd.DataFrame(df_projection_train, index=df_train.index, columns=["x", "y"])
    df_classes = []
    centers = []
    in_class_images = []
    other_class_images = []
    for i in class_idx:
        class_indices = df_train.loc[df_train["label_idx"] == i].loc[:, ["label_idx", "image_idx"]]
        class_projections = df_projections_train.loc[class_indices.index]
        df_class = pd.concat([class_indices, class_projections], axis=1)
        df_classes.append(df_class)
        center = df_class.iloc[:,2:].mean()
        centers.append(center)
        distances = cdist([center], df_class.iloc[:,2:] , metric="euclidean")[0]
        df_class["distance_to_center"] = distances
        df_class = df_class.sort_values(by=["distance_to_center"])
        closest = df_class.iloc[:5, :]
        furthest = df_class.iloc[-5:, :]
        closest_images = torch.cat([dataset[i][0] for i in closest["image_idx"].tolist()], dim=2)
        furthest_images = torch.cat([dataset[i][0] for i in furthest["image_idx"].tolist()], dim=2)
        in_class_images.append(torch.cat((closest_images, furthest_images), dim=1))
    # Find images from other classes that are closest to this class
    df_classes = pd.concat(df_classes, axis=0)
    for i, center in zip(class_idx, centers):
        df_other_classes = df_classes.loc[df_classes["label_idx"] != i].iloc[:,:-1]
        distances = cdist([center], df_other_classes.loc[:,["x", "y"]] , metric="euclidean")[0]
        df_other_classes["distance_to_center"] = distances
        closest = df_other_classes.sort_values(by=["distance_to_center"]).iloc[:5, :]
        closest_images = torch.cat([dataset[i][0] for i in closest["image_idx"].tolist()], dim=2)
        other_class_images.append(closest_images)
    # Save images
    for i in range(len(in_class_images)):
        header_text = Image.new('RGB', (in_class_images[i].shape[2], 20), color=(255, 255, 255))
        draw = ImageDraw.Draw(header_text)
        font = ImageFont.load_default()
        draw.text((10, 5), "TRUE CLASS: " + class_names[class_idx[i]], fill=(0, 0, 0))
        header_text = F.to_tensor(header_text) 
        image = torch.cat([header_text, in_class_images[i], other_class_images[i]], dim=1)
        image = F.to_pil_image(image)
        image.save(join(figs_path, f"close_faraway_closeotherclass_class_{i}.png"))

        
