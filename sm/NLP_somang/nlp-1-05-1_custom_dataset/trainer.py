from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F


class Trainer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        super().__init__()

    def test(self, test_loader, y):
        test_loss = 0
        y_hat = []

        self.model.eval()
        with torch.no_grad():
            for x_i, y_i in test_loader:
                y_hat_i = self.model(x_i)
                loss = F.binary_cross_entropy(y_hat_i, y_i)

                test_loss += loss  # Gradient is already detached.

                y_hat += [y_hat_i]

        test_loss = test_loss / len(test_loader)
        y_hat = torch.cat(y_hat, dim=0)

        print("Test loss: %.4e" % test_loss)

        correct_cnt = (y[2] == (y_hat > .5)).sum()
        total_cnt = float(y[2].size(0))

        print('Test Accuracy: %.4f' % (correct_cnt / total_cnt))

    def train(self, train_loader, valid_loader):
        n_epochs = 10000
        print_interval = 500
        early_stop = 100

        lowest_loss = np.inf
        best_model = None

        lowest_epoch = np.inf

        train_history, valid_history = [], []

        for i in range(n_epochs):
            self.model.train()  # training mode

            train_loss, valid_loss = 0, 0
            y_hat = []

            for x_i, y_i in train_loader:
                y_hat_i = self.model(x_i)   # forward 함수로 hyp 값 구함
                loss = F.binary_cross_entropy(y_hat_i, y_i)  # hyp, y 값으로 cost 계산

                self.optimizer.zero_grad() # 각 tensor 변수 변화도를 0으로 만든다. backward 호출 할 때마다 변화도가 버퍼에 누적되기 때문
                loss.backward() # backpropagation 수행

                self.optimizer.step() # backpropagation 결과를 보고 w,b 값 변경
                train_loss += float(loss)  # This is very important to prevent memory leak.

            train_loss = train_loss / len(train_loader)

            self.model.eval()
            with torch.no_grad(): # 기울기를 변경시키지 않는다, 각 변수들의 변화도를 알 필요가 없다
                valid_loss = 0

                for x_i, y_i in valid_loader:
                    y_hat_i = self.model(x_i)
                    loss = F.binary_cross_entropy(y_hat_i, y_i)

                    valid_loss += float(loss)

                    y_hat += [y_hat_i]

            valid_loss = valid_loss / len(valid_loader)

            train_history += [train_loss]
            valid_history += [valid_loss]

            if (i + 1) % print_interval == 0:
                print('Epoch %d: train loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e' % (
                    i + 1,
                    train_loss,
                    valid_loss,
                    lowest_loss,
                ))

            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                lowest_epoch = i

                best_model = deepcopy(self.model.state_dict())
            else:
                # 최근 100번 동안 cost 가 작아지지 않을 경우 종료
                if early_stop > 0 and lowest_epoch + early_stop < i + 1:
                    print("There is no improvement during last %d epochs." % early_stop)
                    print("lowest_epoch : ",lowest_epoch)
                    print("i : ", i)
                    break

        print("The best validation loss from epoch %d: %.4e" % (lowest_epoch + 1, lowest_loss))
        self.model.load_state_dict(best_model)

        return train_history, valid_history
