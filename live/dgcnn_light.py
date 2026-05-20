"""
DGCNN-Light — Memory-efficient variant for large batch training.
Reduced from the full DGCNN (Wang et al., MIT, 2019):
  - 3 EdgeConv layers instead of 4
  - k=10 neighbors instead of 20
  - Smaller channel widths (256→512 aggregation instead of 512→1024)
  - Smaller FC head (512→256 instead of 1024→512→256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    Find k-nearest neighbors.

    Args:
        x: (B, C, N) point features
        k: number of neighbors

    Returns:
        (B, N, k) indices of k-nearest neighbors
    """
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    _, idx = pairwise_distance.topk(k=k, dim=-1)
    return idx


def get_graph_feature(x: torch.Tensor, k: int = 10, idx: torch.Tensor = None) -> torch.Tensor:
    """
    Construct edge features for EdgeConv.

    Args:
        x: (B, C, N) point features
        k: number of neighbors
        idx: optional precomputed knn indices

    Returns:
        (B, 2*C, N, k) edge features
    """
    batch_size, num_dims, num_points = x.size()
    device = x.device

    if idx is None:
        idx = knn(x, k=k)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)

    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class EdgeConv(nn.Module):
    """EdgeConv layer."""

    def __init__(self, in_channels: int, out_channels: int, k: int = 10):
        super().__init__()
        self.k = k
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature = get_graph_feature(x, k=self.k)
        feature = self.conv(feature)
        feature = feature.max(dim=-1)[0]
        return feature


class DGCNN(nn.Module):
    """
    DGCNN-Light encoder.
    Input: (B, N, 3) point cloud
    Output: (B, embed_dim) embedding
    """

    def __init__(self, embed_dim: int = 256, k: int = 10):
        super().__init__()
        self.k = k
        self.embed_dim = embed_dim

        # 3 EdgeConv layers (removed 4th 128→256 layer)
        self.conv1 = EdgeConv(3, 64, k)
        self.conv2 = EdgeConv(64, 64, k)
        self.conv3 = EdgeConv(64, 128, k)

        # Aggregation (256 input from concat of 64+64+128)
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=1, bias=False),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )

        # Embedding head
        self.fc1 = nn.Linear(512, embed_dim)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.dp1 = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, 3) point cloud

        Returns:
            (B, embed_dim) embedding
        """
        x = x.transpose(1, 2)

        x1 = self.conv1(x)   # (B, 64, N)
        x2 = self.conv2(x1)  # (B, 64, N)
        x3 = self.conv3(x2)  # (B, 128, N)

        x = torch.cat((x1, x2, x3), dim=1)  # (B, 256, N)

        x = self.conv4(x)  # (B, 512, N)

        x = x.max(dim=-1)[0]  # (B, 512)

        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2)
        x = self.dp1(x)

        return x
