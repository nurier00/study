import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from utils import load_mnist
from trainer import Trainer
from model import Autoencoder

from argparse import Namespace


def show_image(x):
    if x.dim() == 1:
        x = x.view(int(x.size(0) ** .5), -1)

    plt.imshow(x, cmap='gray')
    plt.show()


config = {
    'train_ratio': .8,
    'batch_size': 256,
    'n_epochs': 50,
    'verbose': 1,
    'btl_size': 2
}

config = Namespace(**config)

print(f"config : {config}")

train_x, train_y = load_mnist(flatten=True)
test_x, test_y = load_mnist(is_train=False, flatten=True)

train_cnt = int(train_x.size(0) * config.train_ratio)
valid_cnt = train_x.size(0) - train_cnt

indices = torch.randperm(train_x.size(0))

train_x, valid_x = torch.index_select(
    train_x,
    dim=0,
    index=indices
).split([train_cnt, valid_cnt], dim=0)

train_y, valid_y = torch.index_select(
    train_y,
    dim=0,
    index=indices
).split([train_cnt, valid_cnt], dim=0)

print(f"Train_x.shape : {train_x.shape}, Train_y.shape : {train_y.shape}")
# Train_x.shape : torch.Size([48000, 784]), Train_y.shape : torch.Size([48000])

print(f"Valid_x.shape : {valid_x.shape}, Valid_y.shape : {valid_y.shape}")
# Valid_x.shape : torch.Size([12000, 784]), Valid_y.shape : torch.Size([12000])

print(f"Test_x.shape : {test_x.shape}, Test_y.shape : {test_y.shape}")
# Test_x.shape : torch.Size([10000, 784]), Test_y.shape : torch.Size([10000])

model = Autoencoder(btl_size=config.btl_size)
optimizer = optim.Adam(model.parameters())
crit = nn.MSELoss()

trainer = Trainer(model, optimizer, crit)

trainer.train((train_x, train_x), (valid_x, valid_x), config)


# Mean value in each space

with torch.no_grad():
    import random

    index1 = int(random.random() * test_x.size(0))
    index2 = int(random.random() * test_x.size(0))

    print(f"index1 : {index1}, index2 : {index2}")
    # index1 : 9723, index2 : 5849

    z1 = model.encoder(test_x[index1].view(1, -1))
    z2 = model.encoder(test_x[index2].view(1, -1))

    print(f"z1.shape : {z1.shape}, z2.shape : {z2.shape}")
    # z1.shape : torch.Size([1, 2]), z2.shape : torch.Size([1, 2])

    recon = model.decoder((z1 + z2) / 2)

    print(f"recon.shape : {recon.shape}")
    # recon.shape : torch.Size([1, 784])

    recon = recon.squeeze()

    print(f"recon_squeeze.shape : {recon.shape}")
    # recon_squeeze.shape : torch.Size([784])

    show_image(test_x[index1])
    show_image(test_x[index2])
    show_image((test_x[index1] + test_x[index2]) / 2)
    show_image(recon)
