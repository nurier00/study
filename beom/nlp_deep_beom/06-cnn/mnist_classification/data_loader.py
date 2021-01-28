import torch

from torch.utils.data import Dataset, DataLoader

class MnistDataset(Dataset) :
    # torch.utils.data.Dataset 은 추상클래스이다
    # 따라서 함수를 오버라이드 해야한다
    #    __len__  : len(dataset) 호출
    #    __getitem__  : dataset[i] 호출

    def __init__(self, data, labels, flatten=True):
        self.data = data
        self.labels = labels
        self.flatten = flatten

        super().__init__()

    def __len__(self):
        return self.data.size(0)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.flatten:
            x = x.view(-1)

        return x, y


def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST('../data', train=is_train, download=True)

    # dataset = datasets.MNIST(
    #     '../data', train=is_train, download=True,
    #     transform=transforms.Compose([
    #         transforms.ToTensor(),
    #     ]),
    # )

    # x = dataset.data.float() / 255.
    # y = dataset.targets
    #
    # if flatten:
    #     x = x.view(x.size(0), -1)
    #
    # return x, y

    train_input = dataset.train_data.view(-1, 1, 28, 28).float()
    train_target = dataset.train_labels

    return train_input, train_target


def load_mnist_train(is_train=True, flatten=True):
    from torchvision import datasets
    mnist_train_set = datasets.MNIST('../data', train=True, download=True)

    train_input = mnist_train_set.train_data.view(-1, 1, 28, 28).float()
    train_target = mnist_train_set.train_labels
    return train_input, train_target


def load_mnist_test(is_train=True, flatten=True):
    from torchvision import datasets
    mnist_test_set = datasets.MNIST('../data', train=False, download=True)

    test_input = mnist_test_set.test_data.view(-1, 1, 28, 28).float()
    test_target = mnist_test_set.test_labels
    return test_input, test_target


def get_loaders(config):
    ##x, y = load_mnist(is_train=True, flatten=False)
    x, y = load_mnist_train(is_train=True, flatten=False)

    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt

    flatten = True if config.model == 'fc' else False

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0))
    train_x, valid_x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)
    train_y, valid_y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).split([train_cnt, valid_cnt], dim=0)

    train_loader = DataLoader(
        dataset=MnistDataset(train_x, train_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=True,
    )
    valid_loader = DataLoader(
        dataset=MnistDataset(valid_x, valid_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=True,
    )

    ##test_x, test_y = load_mnist(is_train=False, flatten=False)
    test_x, test_y = load_mnist_test(is_train=False, flatten=False)
    test_loader = DataLoader(
        dataset=MnistDataset(test_x, test_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=False,
    )

    return train_loader, valid_loader, test_loader