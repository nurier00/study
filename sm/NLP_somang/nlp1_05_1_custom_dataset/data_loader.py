from torch.utils.data import Dataset, DataLoader


def get_custom_data(x, y):
    batch_size = 128

    train_loader = DataLoader(
        dataset=CustomDataset(x[0], y[0]),
        batch_size=batch_size,
        shuffle=True,   # Allow shuffling only for training set.
    )
    valid_loader = DataLoader(
        dataset=CustomDataset(x[1], y[1]),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        dataset=CustomDataset(x[2], y[2]),
        batch_size=batch_size,
        shuffle=False,
    )

    print("Train %d / Valid %d / Test %d" % (
        len(train_loader.dataset), len(valid_loader.dataset), len(test_loader.dataset),
    ))
    # Train 341 / Valid 113 / Test 115

    return train_loader, valid_loader, test_loader


class CustomDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

        super().__init__()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
