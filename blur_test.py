import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid, save_image
from models import StdLaplacian

thres_range = torch.linspace(1, 10, 100)


def main():
    os.makedirs("output", exist_ok=True)
    model = StdLaplacian()
    dataset = ImageFolder(root="./imgs", transform=ToTensor())
    print(dataset)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    batch = next(iter(loader))
    out = model(batch[0])

    for thres in thres_range:
        ind = out < thres
        if torch.all(ind) or torch.all(~ind):
            print(f"thres {thres} is not good")
            continue
        grid_img = make_grid(batch[0][ind],
                             nrow=8,
                             normalize=True,
                             scale_each=True)
        save_image(grid_img, f"output/thres_{thres}_blur.png")
        grid_img = make_grid(batch[0][~ind],
                             nrow=8,
                             normalize=True,
                             scale_each=True)
        save_image(grid_img, f"output/thres_{thres}_nonblur.png")


if __name__ == "__main__":
    main()
