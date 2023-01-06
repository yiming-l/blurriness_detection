"""PyTorch models for image blurriness assessment."""
import torch
from torch import nn


class StdLaplacian(nn.Module):
    """
    Standard deviation of Laplacian filter.
    """

    def __init__(self):
        super().__init__()
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3,
                                                                  3).float()
        self.kernel = nn.Parameter(k, requires_grad=False)

    def forward(self, img: torch.FloatTensor) -> torch.FloatTensor:
        """compute standard deviation of Laplacian filter.

        Args:
            img (torch.FloatTensor): image tensor with shape (N, C, H, W)

        Returns:
            torch.FloatTensor: standard deviation of Laplacian filter with shape (N,)
        """
        out = img.mul(255).mean(dim=1, keepdim=True)
        out = nn.functional.conv2d(out, self.kernel, padding=0, stride=1)
        out = out.std(dim=(1, 2, 3))
        return out
