from copy import deepcopy

import numpy as np

import torch

class Trainer():
    def __init__(self, model, optimizer, crit):
        self.model = model
        self.optimizer = optimizer
        self.crit = crit

        super().__init__()

    def _train(self, x, y, config):
        self.model.train()

        indices = torch.randperm(x.size(0), device=x.device)

        # torch.index_select(input, dim, index)
        #   : 배열에서 특정값만 조회
        # torch.split(tensor,split_size,dim=0)
        #   : tensor 나눔
        x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
        y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            y_hat_i = self.model(x_i)
            loss_i = self.crit(y_hat_i, y_i.squeeze())

            self.optimizer.zero_grad()
            loss_i.backward()
            self.optimizer.step()

            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

            total_loss += float(loss_i)

        return total_loss / len(x)

    def _validate(self, x, y, config):
        self.model.eval()

        with torch.no_grad():
            indices = torch.randperm(x.size(0), device=x.device)

            x = torch.index_select(x, dim=0, index=indices).split(config.batch_size, dim=0)
            y = torch.index_select(y, dim=0, index=indices).split(config.batch_size, dim=0)

            total_loss = 0
            ## 여기까지는 _train 함수와 동일

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                y_hat_i = self.model(x_i)
                loss_i = self.crit(y_hat_i, y_i.squeeze())

                ## validate 에서는 optimizer 내용이 없다
                # self.optimizer.zero_grad()
                # loss_i.backward()
                # self.optimizer.step()

                if config.verbose >= 2:
                    print("Train Iteration(%d/%d): loss=%.4e" % (i + 1, len(x), float(loss_i)))

                total_loss += float(loss_i)

            return total_loss / len(x)

    # 반복 학습
    def train(self, train_data, valid_data, config):
        # numpy.inf
        #    : (양의) 무한대의 IEEE 754 부동소수점 표현
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs+1):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

            if valid_loss <= lowest_loss :
                lowest_loss = valid_loss;

                # model.state_dict()
                #    : 모델의 매개변수 tensor 계층(사전) 객체
                best_model = deepcopy(self.model.state_dict())

            if epoch_index % 50 == 0 :
                print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                    epoch_index + 1,
                    config.n_epochs,
                    train_loss,
                    valid_loss,
                    lowest_loss,
                ))

        # Restore to best model.
        self.model.load_state_dict(best_model)










