import torch.nn as nn


class DataModel(nn.Module):

    def __init__(self, x, y):
        super().__init__()

        self.x = x
        self.y = y
        self.model = nn.Sequential(
            nn.Linear(x[0].size(-1), 6),
            nn.LeakyReLU(),
            nn.Linear(6, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 4),
            nn.LeakyReLU(),
            nn.Linear(4, 3),
            nn.LeakyReLU(),
            nn.Linear(3, y[0].size(-1)),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.model(x)
        return y
