import torch


def loss_function(recon_x, x, mu, logvar, anneal=1.0):

    bce = torch.mean(torch.sum(recon_x * x))
    kld = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))

    return bce + anneal * kld
