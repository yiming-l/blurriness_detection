import torch
from torch import nn


class StdLaplacian(nn.Module):
    def __init__(self):
        super(StdLaplacian, self).__init__()
        k = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]]).view(1, 1, 3, 3).float()
        self.kernel = nn.Parameter(k, requires_grad=False)

    def forward(self, img: torch.FloatTensor):
        out = img.mul(255).mean(dim=1, keepdim=True)
        out = nn.functional.conv2d(out, self.kernel, padding=0, stride=1)
        out = out.std(dim=(1, 2, 3))
        return out
