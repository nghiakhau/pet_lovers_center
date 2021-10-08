import torch
import torch.nn.functional as F


def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    # bce = F.binary_cross_entropy(recon_x, x)
    recon_x = recon_x.unsqueeze(1)
    x = x.unsqueeze(1)
    mu = mu.unsqueeze(1)
    logvar = logvar.unsqueeze(1)

    bce = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
    kld = -0.5 * torch.mean(
        torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    return bce + anneal * kld
