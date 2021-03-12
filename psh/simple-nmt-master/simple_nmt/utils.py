import torch
from operator import itemgetter

# 학습하는 모델의 변수들을 인자로 받음, 점점 작아지는게 좋다
# 학습 할 때 학습이 잘 되고 있나 참고할 변수
@torch.no_grad()
def get_grad_norm(parameters, norm_type=2):
    # parameters 안에 있는 변수들을 하나씩 p에 담아서 p.grad가 null 아닐 경우 리스트로 만든다.
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    total_norm = 0

    try:
        for p in parameters:
            #print("p.grad.data >> ", p.grad.data)
            total_norm += (p.grad.data**norm_type).sum()    # 학습하면서 변화하는 데이터
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm

# 학습하는 모델의 변수들을 인자로 받음, 점점 커지는게 좋다
@torch.no_grad()
def get_parameter_norm(parameters, norm_type=2):
    total_norm = 0
    try:
        for p in parameters:
            total_norm += (p.data**norm_type).sum()
        total_norm = total_norm ** (1. / norm_type)
    except Exception as e:
        print(e)

    return total_norm


# 문장길이 내림차순으로 정렬
def sort_by_length(x, lengths):
    batch_size = x.size(0)
    x = [x[i] for i in range(batch_size)]
    lengths = [lengths[i] for i in range(batch_size)]
    orders = [i for i in range(batch_size)]

    sorted_tuples = sorted(zip(x, lengths, orders), key=itemgetter(1), reverse=True)
    sorted_x = torch.stack([sorted_tuples[i][0] for i in range(batch_size)])
    sorted_lengths = torch.stack([sorted_tuples[i][1] for i in range(batch_size)])
    sorted_orders = [sorted_tuples[i][2] for i in range(batch_size)]

    return sorted_x, sorted_lengths, sorted_orders


# 전달 받은 특정 orders 에 따라 정렬
def sort_by_order(x, orders):
    batch_size = x.size(0)
    x = [x[i] for i in range(batch_size)]
    
    sorted_tuples = sorted(zip(x, orders), key=itemgetter(1))
    sorted_x = torch.stack([sorted_tuples[i][0] for i in range(batch_size)])

    return sorted_x
