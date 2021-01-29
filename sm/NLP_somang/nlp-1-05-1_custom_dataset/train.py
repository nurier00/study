import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from utils import load_data
from data_loader import get_custom_data
from model import DataModel
from trainer import Trainer


def main():
    # data load
    df = load_data()
    data = torch.from_numpy(df.values).float()

    x = data[:, :10]  # 앞에 10개까지 입력
    y = data[:, -1:]  # 마지막 데이터는 출력

    # Train / Valid / Test ratio
    ratios = [.6, .2, .2]

    train_cnt = int(data.size(0) * ratios[0])
    valid_cnt = int(data.size(0) * ratios[1])
    test_cnt = data.size(0) - train_cnt - valid_cnt
    cnts = [train_cnt, valid_cnt, test_cnt]

    # 랜덤하게 데이터 섞은 후 각 cnt 별로 분류
    indices = torch.randperm(data.size(0))

    x = torch.index_select(x, dim=0, index=indices)
    y = torch.index_select(y, dim=0, index=indices)

    x = x.split(cnts, dim=0)
    y = y.split(cnts, dim=0)

    train_loader, valid_loader, test_loader = get_custom_data(x, y)

    print("Train %d / Valid %d / Test %d samples." % (
        len(train_loader.dataset),
        len(valid_loader.dataset),
        len(test_loader.dataset),
    ))

    model = DataModel(x, y)
    print(model)
    optimizer = optim.Adam(model.parameters())

    trainer = Trainer(model, optimizer)
    train_history, valid_history = trainer.train(train_loader, valid_loader)

    #
    plot_from = 2

    plt.figure(figsize=(20, 10))
    plt.grid(True)
    plt.title("Train / Valid Loss History")
    plt.plot(
        range(plot_from, len(train_history)), train_history[plot_from:],
        range(plot_from, len(valid_history)), valid_history[plot_from:],
    )
    plt.yscale('log')
    plt.show()

    trainer.test(test_loader, y)


if __name__ == '__main__':
    main()
