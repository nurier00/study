import numpy as np
import matplotlib.pyplot as plt
import random

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


if __name__ == '__main__':
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

    print(f"Train x.shape : {train_x.shape}, Train y.shape : {train_y.shape}")
    print(f"Valid x.shape : {valid_x.shape}, Valid y.shape : {valid_y.shape}")
    print(f"Test x.shape : {test_x.shape}, Test y.shape : {test_y.shape}")

    model = Autoencoder(btl_size=config.btl_size)
    optimizer = optim.Adam(model.parameters())
    crit = nn.MSELoss()

    trainer = Trainer(model, optimizer, crit)
    trainer.train((train_x, train_x), (valid_x, valid_x), config)

    # Test
    with torch.no_grad():
        index = int(random.random() * test_x.size(0))
        recon = model(test_x[index].view(1, -1)).squeeze()

        show_image(test_x[index])
        show_image(recon)

    # Latent Space
    if config.btl_size == 2:
        color_map = [
            'brown', 'red', 'orange', 'yellow', 'green',
            'blue', 'navy', 'purple', 'gray', 'black',
        ]
        plt.figure(figsize=(20, 10))
        with torch.no_grad():
            latents = model.encoder(test_x[:1000])

            for i in range(10):
                target_latents = latents[test_y[:1000] == i]
                target_y = test_y[:1000][test_y[:1000] == i]
                plt.scatter(target_latents[:, 0],
                            target_latents[:, 1],
                            marker='o',
                            color=color_map[i],
                            label=i)

            plt.legend()
            plt.grid(axis='both')
            plt.show()

    if config.btl_size == 2:
        min_range, max_range = -2., 2.
        n = 20
        step = (max_range - min_range) / float(n)

        with torch.no_grad():
            lines = []

            for v1 in np.arange(min_range, max_range, step):
                z = torch.stack([
                    torch.FloatTensor([v1] * n),
                    torch.FloatTensor([v2 for v2 in np.arange(min_range, max_range, step)]),
                ], dim=-1)

                line = torch.clamp(model.decoder(z).view(n, 28, 28), 0, 1)
                line = torch.cat([line[i] for i in range(n - 1, 0, -1)], dim=0)
                lines += [line]

            lines = torch.cat(lines, dim=-1)
            plt.figure(figsize=(20, 20))
            show_image(lines)
