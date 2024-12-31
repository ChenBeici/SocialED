# -*- coding: utf-8 -*-
"""Loss functions and metrics for training social event detection models."""

import torch.nn as nn
import torch.nn.functional as F
import torch

def common_loss(emb1, emb2):
    """Calculate common loss between two embeddings.
    
    Parameters
    ----------
    emb1 : torch.Tensor
        First embedding tensor
    emb2 : torch.Tensor
        Second embedding tensor
        
    Returns
    -------
    cost : torch.Tensor
        The computed common loss
    """
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2)**2)
    return cost

def EUC_loss(alpha,u,true_labels,e):
    """Calculate Evidence-based Uncertainty Classification loss.
    
    Parameters
    ----------
    alpha : torch.Tensor
        Dirichlet distribution parameters
    u : torch.Tensor
        Uncertainty values
    true_labels : torch.Tensor
        Ground truth labels
    e : int
        Current epoch number
        
    Returns
    -------
    loss : torch.Tensor
        The computed EUC loss
    """
    _, pred_label = torch.max(alpha, 1)
    true_indices = torch.where(pred_label == true_labels)
    false_indices = torch.where(pred_label != true_labels)
    S = torch.sum(alpha, dim=1, keepdim=True)
    p, _ = torch.max(alpha / S, 1)
    a = -0.01 * torch.exp(-(e + 1) / 10 * torch.log(torch.FloatTensor([0.01]))).cuda()
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor((e+1) / 10, dtype=torch.float32),
    )
    EUC_loss = -annealing_coef * torch.sum((p[true_indices]*(torch.log(1.000000001 - u[true_indices]).squeeze(
        -1)))) # -(1-annealing_coef)*torch.sum(((1-p[false_indices])*(torch.log(u[false_indices]).squeeze(-1))))

    return EUC_loss

def loglikelihood_loss(y, alpha, device):
    """Compute log-likelihood loss.
    
    Parameters
    ----------
    y : torch.Tensor
        Ground truth labels
    alpha : torch.Tensor
        Dirichlet distribution parameters
    device : str
        Device to use for computation
        
    Returns
    -------
    torch.Tensor
        Log-likelihood loss
    """
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


class OnlineTripletLoss(nn.Module):
    """Online Triplets loss module.
    
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that takes 
    embeddings and targets and returns indices of triplets.

    Parameters
    ----------
    margin : float
        Margin for triplet loss
    triplet_selector : TripletSelector
        Selector object for generating triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target, rd, peer_embeddings=None):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        if peer_embeddings != None:
            peer_ap_distances = (peer_embeddings[triplets[:, 0]] - peer_embeddings[triplets[:, 1]]).pow(2).sum(1)
            peer_an_distances = (peer_embeddings[triplets[:, 0]] - peer_embeddings[triplets[:, 2]]).pow(2).sum(1)
            kd_ap_losses = F.relu(-peer_ap_distances + ap_distances)
            kd_an_losses = F.relu(-an_distances + peer_an_distances)
            print("losses.mean():", losses.mean(), "ap_mean:", kd_ap_losses.mean(), "an_mean:", kd_an_losses.mean())
            return losses.mean() + rd * kd_ap_losses.mean() + rd * kd_an_losses.mean(), len(triplets)
        else:
            return losses.mean(), len(triplets)



def kl_divergence(alpha, num_classes, device):
    """Compute KL divergence for Dirichlet distributions.
    
    Parameters
    ----------
    alpha : torch.Tensor
        Dirichlet distribution parameters
    num_classes : int
        Number of classes
    device : str
        Device to use for computation
        
    Returns
    -------
    torch.Tensor
        KL divergence loss
    """
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device):
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div

def edl_loss(func, y, true_labels, alpha, epoch_num, num_classes, annealing_step, device):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor((epoch_num+1) / 10, dtype=torch.float32),
    )

    _, pred_label = torch.max(alpha, 1)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    print("kl_div:",1*torch.mean(kl_div))
    print("A:",20*torch.mean(A))

    return 20*A + 1*kl_div

