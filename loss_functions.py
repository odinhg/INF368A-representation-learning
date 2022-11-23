import torch
import torch.nn as nn

class TripletLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = torch.tensor(margin)
    
    def forward(self, embeddings, labels, negative_policy="semi-hard", positive_policy="easy"):
        device = embeddings.device
        self.margin.to(device)
        embedding_dimension = embeddings[0].shape[0]
        classes_in_batch = torch.unique(labels)
        dist_mat = torch.cdist(embeddings, embeddings)**2
        triplet_loss = torch.tensor(0.0, device=device, requires_grad=True) 
        number_of_triplets_mined = 0
        # Online Triplet Mining
        for c in classes_in_batch:
            ap_indices = (labels == c).nonzero()
            if embeddings[ap_indices].reshape(-1, embedding_dimension).shape[0] < 2:
                continue
            n_indices = (labels != c).nonzero()
            for a_idx in ap_indices: 
                dists_an = dist_mat[a_idx, n_indices]
                
                dists_ap = dist_mat[a_idx, ap_indices]
                dists_ap = dists_ap[dists_ap.nonzero(as_tuple=True)]
                if positive_policy == "easy":
                    dist_ap = torch.min(dists_ap)
                else:
                    dist_ap = torch.max(dists_ap)

                if negative_policy == "semi-hard":
                    mined_indices = torch.logical_and((dists_an > dist_ap), (dists_an < dist_ap + self.margin)).nonzero()
                else: # Mine hard triplets
                    mined_indices = (dists_an < dist_ap).nonzero()

                if mined_indices.numel() == 0: # In case we don't mine any triplets...
                    dist_an = torch.min(dists_an) # Just add the closest negative to avoid nan
                else:
                    dist_an = dist_mat[a_idx, mined_indices[0,0]]

                loss = nn.functional.relu(dist_ap - dist_an + self.margin)
                triplet_loss = triplet_loss + loss
                number_of_triplets_mined += 1

        return triplet_loss / number_of_triplets_mined

class AngularMarginLoss(nn.Module):
    # Simple implementation of Angular Margin Loss for ArcFace
    def __init__(self, m, s, number_of_classes):
        super().__init__()
        self.m = torch.tensor(m)
        self.s = torch.tensor(s)
        self.number_of_classes = number_of_classes
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, embeddings, weights, labels):
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1) #L2-normalize embeddings
        weights = nn.functional.normalize(weights, p=2, dim=1) #L2-normalize weights
        device = embeddings.device
        self.m.to(device)
        self.s.to(device)
        cos_theta = torch.matmul(embeddings, weights.t())
        theta = torch.acos(cos_theta)
        # Using one-hot to add margin only to ground truth
        margin = self.m * torch.nn.functional.one_hot(labels, num_classes=self.number_of_classes)
        logits = self.s * torch.cos(theta + margin)
        loss = self.cross_entropy_loss(logits, labels)
        return loss

class NTXentLoss(nn.Module):
    # Normalized Temperature-scaled Cross Entropy Loss for SimCLR
    def __init__(self, t):
        super().__init__()
        self.t = torch.tensor(t) # Temperature hyper-parameter

    def forward(self, z1, z2):
        device = z1.device
        self.t.to(device)
        # Normalize embeddings
        z1 = nn.functional.normalize(z1, p=2, dim=1)
        z2 = nn.functional.normalize(z2, p=2, dim=1)
        batch_size = z1.shape[0]
        # Interleave embeddings from the two views (odd / even)
        z = torch.stack((z1, z2), dim=1).view(2*batch_size, -1)
        # Compute cosine similarity, divide by temperature and exponentiate
        similarities = torch.matmul(z, z.t())
        similarities = similarities / self.t
        exp_sim = torch.exp(similarities)
        # Computing the denominator of l(i,j) 
        mask = 1 - torch.eye(2*batch_size, device=device)
        denominator = torch.sum(exp_sim * mask, dim=1)
        denominator = denominator.expand(batch_size*2, -1).t()
        # l(i,j) = L[i,j]
        L = -torch.log(torch.div(exp_sim, denominator))
        # Only sum the entries l(2k, 2k+1) and l(2k+1, 2k)
        mask = torch.block_diag(*(1 - torch.eye(2, device=device)).unsqueeze(0).repeat(batch_size,1,1))
        loss = torch.sum(L * mask) / (2 * batch_size)
        return loss
