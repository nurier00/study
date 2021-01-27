import argparse
import torch
import torch.nn

import sys
import numpy as np
import matplotlib.pyplot as plt

from mnist_classification.data_loader import load_mnist, load_mnist_test

from mnist_classification.models.fc_model import FullyConnectedClassifier
from mnist_classification.models.cnn_model import ConvolutionalClassifier

#model_fn = "./model_fc.pth"
p = argparse.ArgumentParser()
p.add_argument('--model_fn', required=True)
config = p.parse_args()
model_fn = config.model_fn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def load(fn, device):
    d = torch.load(fn, map_location=device)
    return d['config'], d['model']


def plot(x, y_hat):
    for i in range(x.size(0)):
        img = (np.array(x[i].detach().cpu(), dtype='float')).reshape(28,28)

        print("Predict:", float(torch.argmax(y_hat[i], dim=-1)))

        plt.imshow(img, cmap='gray')
        plt.show()


def test(model, x, y, to_be_shown=True):
    model.eval()

    with torch.no_grad():
        y_hat = model(x)

        correct_cnt = (y.squeeze() == torch.argmax(y_hat, dim=-1)).sum()
        total_cnt = float(x.size(0))

        accuracy = correct_cnt / total_cnt
        print("Accuracy: %.4f" % accuracy)

        if to_be_shown:
            plot(x, y_hat)


from train import get_model

train_config, state_dict = load(model_fn, device)

model = get_model(train_config).to(device)
model.load_state_dict(state_dict)

print(model)

# Load MNIST test set.
#x, y = load_mnist(is_train=False, flatten=True if train_config.model == 'fc' else False)
x, y = load_mnist_test(is_train=False, flatten=True if train_config.model == 'fc' else False)

x, y = x.to(device), y.to(device)

test(model, x[:20], y[:20], to_be_shown=True)


# prediction

####  python prediction.py --model_fn=./model_fc.pth

####  python prediction.py --model_fn=./model_cnn_cpu.pth

####  python prediction.py --model_fn=./model_fc_gpu.pth

####  python prediction.py --model_fn=./model_cnn_gpu.pth

