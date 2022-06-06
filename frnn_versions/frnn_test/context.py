import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch as t
import torch.utils.data
from torchvision import transforms
import torchvision

torch.manual_seed(7)
torch.cuda.manual_seed_all(7)


def get_objects(data):
    device = data.device

    D1 = t.arange(data.size(2), dtype=t.int)
    D2 = t.arange(data.size(3), dtype=t.int)

    gridy, gridx = t.meshgrid(D1,D2)
    gridy = gridy.unsqueeze(0).unsqueeze(0).repeat(data.size(0),1,1,1).float().to(device)
    gridx = gridx.unsqueeze(0).unsqueeze(0).repeat(data.size(0),1,1,1).float().to(device)
    zeros = t.zeros_like(gridx)
    ones =  t.ones_like(gridx)

    angles = t.zeros_like(gridx)
    geom = t.cat([gridy, gridx, zeros, angles, ones, zeros], 1)

    imgid = t.arange(data.size(0), device=device).unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1,1,data.size(2), data.size(3))

    texture =   data.permute([0,2,3,1])
    geom =      geom.permute([0,2,3,1])
    imgid =   imgid.permute([0,2,3,1])

    texture =   texture.reshape(-1,texture.size(3))
    geom =      geom.reshape(-1,geom.size(3))
    imgid =   imgid.reshape(-1,imgid.size(3))
    imgid =   imgid.squeeze(1)

    return texture.to(device), geom.to(device), imgid.to(device)

def get_test_loader(batch_size, img_size=32):
    workers = 4

    tran_list = list()
    tran_list.append(transforms.Resize([img_size, img_size]))
    tran_list.append(transforms.ToTensor())
    transform = transforms.Compose(tran_list)

    dataset = torchvision.datasets.CIFAR10(root='~/datasets/brainkit/', train=True, download=True, transform=transform)

    if isinstance(batch_size, torch.Tensor): batch_size = batch_size.item()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                         num_workers=workers, pin_memory=False)
    return loader
