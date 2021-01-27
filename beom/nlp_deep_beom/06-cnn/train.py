import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from mnist_classification.data_loader import get_loaders
from mnist_classification.trainer import Trainer

from mnist_classification.models.fc_model import FullyConnectedClassifier
from mnist_classification.models.cnn_model import ConvolutionalClassifier


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', default='./model.pth')
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.8)

    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--model', type=str, default='fc')

    config = p.parse_args()

    return config


def get_model(config):
    if config.model == 'fc':
        model = FullyConnectedClassifier(28**2, 10)
        ##config.model_fn = './model_fc.pth'
    elif config.model == 'cnn':
        model = ConvolutionalClassifier(10)
        ##config.model_fn = './model_cnn.pth'
    else:
        raise NotImplementedError('You need to specify model name.')

    return model


def main(config):
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)

    print('device : ', device)

    train_loader, valid_loader, test_loader = get_loaders(config)

    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))
    print("Test:", len(test_loader.dataset))

    model = get_model(config).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.NLLLoss()

    if config.verbose >= 2:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader)

if __name__ == '__main__':
    print(torch.version)
    config = define_argparser()
    main(config)


############################################################
# train

## fc model - cpu
####  python train.py --model_fn=./model_fc_cpu.pth --gpu_id=-1 --model=fc

## cnn model - cpu
####  python train.py --model_fn=./model_cnn_cpu.pth --gpu_id=-1 --model=cnn
############################################################

############################################################
# gpu mode
## install pycuda
## install torch-vision

# visual stdio 2019 설치 > 기본설치
#
#    https://visualstudio.microsoft.com/ko/vs/
#
# Nvidia cuda 설치 ( 10.2 v 설치 )
#    https://developer.nvidia.com/cuda-downloads
#    https://developer.nvidia.com/cuda-10.2-download-archive?target_os=Windows&target_arch=x86_64
# ( 연동 표 : https://conda.anaconda.org/pytorch/win-64 )
#
# Nvidia cuDNN 설치 (회원필요)
#    https://developer.nvidia.com/CUDNN
#    Download cuDNN v7.6.5 (November 18th, 2019), for CUDA 10. 2
#      -> cuda 설치경로에 압축 해제
#          C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2


# Test :
#   C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.2\1_Utilities\deviceQuery

# pytorch 재설치 - gpu 버전
#     https://pytorch.org/get-started/locally/
#     pip install torch===1.7.1 torchvision===0.8.2 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html



## error : cuda error
##      -> https://developer.nvidia.com/cuda-toolkit
## error(X) : Microsoft Visual C++ 14.0 is required. Get it with "Build Tools for Visual Studio"
##      -> https://visualstudio.microsoft.com/ko/vs/older-downloads/
##         재배포 가능 패키지 및 빌드 도구 -> Microsoft Build Tools 2015 업데이트 3
## error : pip version
##      -> python -m pip install --upgrade pip
## cuda.h 파일 못찾는경우 재부팅 필요
############################################################


############################################################
# ## fc model - gpu

# conda install pytorch=1.7 torchvision cudatoolkit=9.2 -c pytorch

## --gpu_id=0  -> GPU Index( -1 : CPU )
## fc model - gpu
####  python train.py --model_fn=./model_fc_gpu.pth --gpu_id=0 --model=fc

## cnn model - gpu
####  python train.py --model_fn=./model_cnn_gpu.pth --gpu_id=0 --model=cnn
