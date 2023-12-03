import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd


class RBF(nn.Module):
    """A simple RBF kernel."""
    def __init__(self, sigma=None):
        super(RBF, self).__init__()
        self.sigma = sigma

    def forward(self, X, Y):
        """
        Compute the RBF kernel between X and Y.
        :param X:
        :param Y:
        :return:
        """
        if len(X.shape) == 1:
            X = X.unsqueeze(-1)
        if len(Y.shape) == 1:
            Y = Y.unsqueeze(-1)
        XX = X.matmul(X.t())
        XY = X.matmul(Y.t())
        YY = Y.matmul(Y.t())

        dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

        # Apply the median heuristic (PyTorch does not give true median)
        if self.sigma is None:
            np_dnorm2 = dnorm2.detach().cpu().numpy()
            h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
            sigma = np.sqrt(h).item()
        else:
            sigma = self.sigma

        gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
        K_XY = (-gamma * dnorm2).exp()

        return K_XY


class SVGD(object):
    def __init__(self, P, K, optimizer):
        self.P = P
        self.K = K
        self.optim = optimizer

    def phi(self, X):
        X = X.detach().requires_grad_(True)

        log_prob = self.P.log_prob(X)
        score_func = autograd.grad(log_prob.sum(), X)[0]

        K_XX = self.K(X, X.detach())
        grad_K = -autograd.grad(K_XX.sum(), X)[0]
        phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)

        return phi

    def step(self, X):
        self.optim.zero_grad()
        X.grad = -self.phi(X)
        self.optim.step()


class CostProbWrapper(object):
    """
    A wrapper for a cost function that makes it a probability distribution.
    """
    def __init__(self, cost_fnc, surface_pointcloud, free_pointcloud, latent_size=256):
        self.cost = cost_fnc
        self.latent_size = latent_size # (..., N, 3)
        self.surface_pointcloud = surface_pointcloud # (..., N, 3)
        self.free_pointcloud = free_pointcloud
        self.loss_value = None

    def compute_cost(self, X):
        latent, pose = self._split_X(X)
        loss, loss_ind = self.cost(latent, pose, self.surface_pointcloud, self.free_pointcloud)
        self.loss_value = loss
        return loss

    def _split_X(self, X):
        batch_size = self.surface_pointcloud.shape[:-2]
        latent = X[:, :self.latent_size].reshape(batch_size + (-1,))
        pose = X[:, self.latent_size:].reshape(batch_size + (-1,))
        return latent, pose

    def log_prob(self, X):
        # The cost is the negative log probability.
        # We assume that p=e^(-cost) is a valid probability distribution.
        # therefore, log p = -cost
        loss = self.compute_cost(X)
        logp = -loss
        return logp


