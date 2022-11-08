import torch
import torch.nn.functional as F
from torch import nn as nn


def cp_lossfn(x, distance="mse"):
    """x: NxTx..."""
    x_mean = torch.mean(x, dim=1)
    dist_fun = {
        "mse": F.mse_loss,
    }[distance]
    loss_cp = torch.mean(
        dist_fun(x[:, 0],
                 x_mean) + \
        dist_fun(x[:, 1],
                 x_mean) + \
        dist_fun(x[:, 2],
                 x_mean) + \
        dist_fun(x[:, 3],
                 x_mean)
    )
    return loss_cp


class FlipTransformer(nn.Module):
    def forward(self, x):
        """x: NxCxHxW"""

        xh = torch.flip(x, dims=[2])
        xw = torch.flip(x, dims=[3])
        xhw = torch.flip(x, dims=[2, 3])

        x = torch.stack([x, xh, xw, xhw], dim=1)  # NxTxCxHxW

        return x


class ConsistencyPriorModel(nn.Module):
    """
    Similar to [Bortsova 2019] adapted for classification.

    [Bortsova 2019] Bortsova, Gerda, et al. "Semi-supervised medical image segmentation via learning consistency under transformations." International Conference on Medical Image Computing and Computer-Assisted Intervention. Springer, Cham, 2019.
    """
    def __init__(self, network):
        super().__init__()
        self.transformer = FlipTransformer()
        self.network = network

    def forward(self, x):
        """x: NxCxHxW"""
        n, c, h, w = x.shape
        t = 4  # Number of transformed versions

        x_transformed = self.transformer(x)  # NTCHW

        logits = self.network(x_transformed.view(n * t, c, h, w))
        logits = logits.view(n, t, -1)

        return logits

    def forward_avg(self, x):
        """
        Average embeddings of different transformations.
        x: NxCxHxW
        """
        n, c, h, w = x.shape
        t = 4  # Number of transformed versions

        x_transformed = self.transformer(x)  # NTCHW

        logits = self.network(x_transformed.view(n * t, c, h, w))
        logits = logits.view(n, t, -1)
        logits_avg = logits.mean(dim=1)

        return logits_avg
