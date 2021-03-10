from operator import itemgetter

import torch
import torch.nn as nn

import simple_nmt.data_loader as data_loader

LENGTH_PENALTY = .2
MIN_LENGTH = 5


class SingleBeamSearchBoard():

    def __init__(
        self,
        device,
        prev_status_config,
        beam_size=5,
        max_length=255,
    ):
        self.beam_size = beam_size
        self.max_length = max_length

        # To put data to same device.
        self.device = device

        # 각 time-step 별 추론된 단어를 저장. 최초에는 <BOS> 토큰이 들어있다.
        self.word_indice = [torch.LongTensor(beam_size).zero_().to(self.device) + data_loader.BOS]
        # word_indice : [tensor([2, 2, 2])]     =>  2 = BOS

        # 각 time-step 별 선택된 단어인덱스가 포함된 beam 인덱스를 저장. 최초에는 -1 이 들어가 있음
        self.beam_indice = [torch.LongTensor(beam_size).zero_().to(self.device) - 1]
        # beam_indice : [tensor([-1, -1, -1])]

        # 각 beam 별 누적확률값이 저장되어 있음.
        self.cumulative_probs = [torch.FloatTensor([.0] + [-float('inf')] * (beam_size - 1)).to(self.device)]
        # cumulative_probs = [tensor([0., -inf, -inf])]

        # 1 if it is done else 0
        # 문장이 끝날 경우 1, 끝나지 않았을 경우 0 으로 처리 - boolean -> 1 = true, 0 = false, 최초에는 모두 false 값이 들어있음
        # 즉, <PAD> 마스크와 동일한 효과를 내줌
        self.masks = [torch.BoolTensor(beam_size).zero_().to(self.device)]
        # masks = [tensor([False, False, False])]

        # 모든 time-step 의 hidden 값을 저장하지 않고 가장 마지막 time-step, 즉, 현재 time-step 에 대한 값만 저장
        # 최초에는 None
        self.prev_status = {}
        self.batch_dims = {}
        for prev_status_name, each_config in prev_status_config.items():
            init_status = each_config['init_status']
            batch_dim_index = each_config['batch_dim_index']
            if init_status is not None:
                self.prev_status[prev_status_name] = torch.cat([init_status] * beam_size,
                                                               dim=batch_dim_index)
            else:
                self.prev_status[prev_status_name] = None
            self.batch_dims[prev_status_name] = batch_dim_index

        self.current_time_step = 0
        self.done_cnt = 0

    # 문장 길이가 짧을수록 더 큰 확률을 가지고 있고, 문장 길이가 길수록 작은 확률을 가지고 있기 때문에,
    # 이 차이를 완화시켜 계산해주기 위해 짧은 문장에는 보다 큰 페널티를, 긴 문장에는 보다 작은 페널티를 곱해서 계산
    def get_length_penalty(
        self,
        length,
        alpha=LENGTH_PENALTY,
        min_length=MIN_LENGTH,
    ):
        # Calculate length-penalty,
        # because shorter sentence usually have bigger probability.
        # In fact, we represent this as log-probability, which is negative value.
        # Thus, we need to multiply bigger penalty for shorter one.
        p = ((min_length + 1) / (min_length + length))**alpha

        return p

    # 문장이 끝나고 <EOS> 토큰이 나오면 1 반환
    def is_done(self):
        # Return 1, if we had EOS more than 'beam_size'-times.
        if self.done_cnt >= self.beam_size:
            return 1
        return 0

    # 현재 time-step 의 y_hat 값 (beam_size 만큼) 과 prev_status 를 반환
    def get_batch(self):
        y_hat = self.word_indice[-1].unsqueeze(-1)
        # |y_hat| = (beam_size, 1)

        # word indice : [tensor([2, 2, 2])]
        # y_hat : tensor([[2], [2], [2]])    -> 단어 인덱스로 들어가 있음 / 인덱스 = 2 => <BOS> (최초 y_hat)
        # y_hat shape : torch.Size([3, 1]) => [beam_size, 1]

        # if model != transformer:
        #     |hidden| = |cell| = (n_layers, beam_size, hidden_size)
        #     |h_t_tilde| = (beam_size, 1, hidden_size) or None
        #
        #     hidden_state : torch.Size([4, 3, 768])    => [n_layers, beam_size, hidden_size]
        #     cell_state : torch.Size([4, 3, 768])      => [n_layers, beam_size, hidden_size]
        #     h_t_1_tilde : None -> torch.Size([3, 1, 768])    => [beam_size, 1, hidden_size]
        #
        # else:
        #     |prev_state_i| = (beam_size, length, hidden_size),
        #     where i is an index of layer.
        #
        #     n_layer = 4 일 경우
        #     prev_status : ['prev_state_0', 'prev_state_1', 'prev_state_2', 'prev_state_3', 'prev_state_4']
        #     prev_status 0 : torch.Size([3, 1, 768])   => [beam_size, 1, hidden_size]
        #     prev_status 1 : torch.Size([3, 1, 768])   => [beam_size, 1, hidden_size]
        #     prev_status 2 : torch.Size([3, 1, 768])   => [beam_size, 1, hidden_size]
        #     prev_status 3 : torch.Size([3, 1, 768])   => [beam_size, 1, hidden_size]
        #     prev_status 4 : torch.Size([3, 1, 768])   => [beam_size, 1, hidden_size]

        return y_hat, self.prev_status

    # @profile
    def collect_result(self, y_hat, prev_status):
        # y_hat shape: torch.Size([3, 1, 6249])
        # |y_hat| = (beam_size, 1, output_size)

        # prev_status is a dict, which has following keys:
        # if model != transformer:
        #     |hidden| = |cell| = (n_layers, beam_size, hidden_size)
        #     |h_t_tilde| = (beam_size, 1, hidden_size)
        # else:
        #     |prev_state_i| = (beam_size, length, hidden_size),
        #     where i is an index of layer.

        output_size = y_hat.size(-1)

        self.current_time_step += 1
        # print(f"current_time_step : {self.current_time_step}")

        # Calculate cumulative log-probability.
        # First, fill -inf value to last cumulative probability, if the beam is already finished.
        # Second, expand -inf filled cumulative probability to fit to 'y_hat'.
        # (beam_size) --> (beam_size, 1, 1) --> (beam_size, 1, output_size) = |y_hat|
        # Third, add expanded cumulative probability to 'y_hat'

        # masks :       [tensor([False, False, False])] => tensor([3]) = beam_size
        # cumulative_probs : [tensor([0., -inf, -inf])] => tensor([3]) = beam_size

        # 누적확률 텐서에 마스크를 씌우고 y_hat 과의 계산을 위해 shape 을 맞춰준다
        cumulative_prob = self.cumulative_probs[-1].masked_fill_(self.masks[-1], -float('inf'))
        cumulative_prob = y_hat + cumulative_prob.view(-1, 1, 1).expand(self.beam_size, 1, output_size)
        # |cumulative_prob| = (beam_size, 1, output_size)
        # cumulative_prob : tensor([[[-394.6682,-402.9751,-399.8368,...,-382.3197,-447.5492,-281.3903]],
        # [[-inf,-inf,-inf,...,-inf,-inf,-inf]], [[-inf,-inf,-inf,...,-inf,-inf,-inf]]])

        # Now, we have new top log-probability and its index.
        # We picked top index as many as 'beam_size'.
        # Be aware that we picked top-k from whole batch through 'view(-1)'.

        # y_hat 에 누적확률을 계산한 뒤, 모두 모아서 정렬시킨뒤 k (=beam_size) 만큼 추출
        # .topk() 보다 .sort() 를 사용하는 것이 더 빠름

        # Following lines are using torch.topk, which is slower than torch.sort.
        # top_log_prob, top_indice = torch.topk(
        #     cumulative_prob.view(-1), # (beam_size * output_size,)
        #     self.beam_size,
        #     dim=-1,
        # )

        # Following lines are using torch.sort, instead of using torch.topk.
        # 누적확률 값과 해당 인덱스를 내림차순으로 정렬시킨다
        top_log_prob, top_indice = cumulative_prob.view(-1).sort(descending=True)
        # top_log_prob : tensor([-2.6131,-3.6799,-4.5126,...,-inf,-inf,-inf]),
        # top_indice : tensor([4,3,35,...,40112,40113,40114])

        # 누적확률과 인덱스를 각각 k (=beam_size) 개 만큼만 남도록 잘라준다
        # 이 때, 남은 top_log_prob, top_indice 가 이번 time-step 에서 뽑힌 k 개만큼의 y_hat 이 된다
        top_log_prob, top_indice = top_log_prob[:self.beam_size], top_indice[:self.beam_size]
        # top_log_prob: tensor([-2.6131, -3.6799, -4.5126]), |top_log_prob| = (beam_size,)
        # top_indice: tensor([4, 3, 35]),                    |top_indice| = (beam_size,)

        # Because we picked from whole batch, original word index should be calculated again.
        self.word_indice += [top_indice.fmod(output_size)]
        # .fmod() : 부동소수점 연산 -> 나머지 연산(%)과 동일하되 +,- 부호를 그대로 가져옴
        # word_indice : [tensor([2,2,2])] -> [tensor([2,2,2]),tensor([4,3,35])]
        # word_indice 에 현재 time-step 의 y_hat 추론값(인덱스값)이 누적되어 들어감

        # Also, we can get an index of beam, which has top-k log-probability search result.
        self.beam_indice += [top_indice.div(float(output_size)).long()]
        # beam_indice : [tensor([-1,-1,-1]),tensor([0,0,0])] -> [tensor([-1,-1,-1]),tensor([0,0,0]),tensor([0,0,2])] ...
        # beam_indice 에는 뽑힌 추론값이 속해있는 k개로 나뉜 빔의 인덱스 값이 들어간다. 최초에 <BOS>는 모두 -1 로 셋팅되어 들어감

        # Add results to history boards.
        self.cumulative_probs += [top_log_prob]
        # cumulative_probs : [tensor([0.,-inf,-inf]),tensor([-2.6131,-inf,-4.5125]),tensor([-5.2262,-inf,-7.1255])]

        self.masks += [torch.eq(self.word_indice[-1], data_loader.EOS)]     # Set finish mask if we got EOS.
        # masks : [tensor([False, False, False]), tensor([False, True, False]), tensor([False, True, False])]
        # 문장이 끝나고 <EOS> 가 들어가게 되면 True 값으로 들어감

        # Calculate a number of finished beams.
        self.done_cnt += self.masks[-1].float().sum()

        # In beam search procedure, we only need to memorize latest status.
        # For seq2seq, it would be latest hidden and cell state, and h_t_tilde.
        # The problem is hidden(or cell) state and h_t_tilde has different dimension order.
        # In other words, a dimension for batch index is different.
        # Therefore self.batch_dims stores the dimension index for batch index.
        # For transformer, latest status is each layer's decoder output from the beginning.
        # Unlike seq2seq, transformer has to memorize every previous output for attention operation.
        for prev_status_name, prev_status in prev_status.items():
            self.prev_status[prev_status_name] = torch.index_select(
                prev_status,
                dim=self.batch_dims[prev_status_name],
                index=self.beam_indice[-1]
            ).contiguous()

    def get_n_best(self, n=1, length_penalty=.2):
        sentences, probs, founds = [], [], []

        for t in range(len(self.word_indice)):  # for each time-step,
            for b in range(self.beam_size):     # for each beam,
                if self.masks[t][b] == 1:       # 현재 time-step 에 EOS 가 있을 경우,
                    # Take a record of penaltified log-proability.
                    probs += [self.cumulative_probs[t][b] * self.get_length_penalty(t, alpha=length_penalty)]
                    founds += [(t, b)]

                    # probs : [tensor(-inf), tensor(-7.3420), tensor(-7.4389)]
                    # founds : [(1, 1), (2, 0), (2, 2)] => [(time-step, beam_index)]

        # Also, collect log-probability from last time-step, for the case of EOS is not shown.
        # 문장이 아직 끝나지 않았을 경우 (<EOS> 토큰이 나오지 않은 경우), 마지막 time-step 의 누적확률을 수집
        for b in range(self.beam_size):
            if self.cumulative_probs[-1][b] != -float('inf'):   # If this beam does not have EOS,
                # cumulative_probs : [tensor([0., -inf, -inf]), tensor([-4.4934, -inf, -4.8619]),
                #                     tensor([-8.8338, -8.8975, -8.9504])]
                # founds : [(1, 1), (2, 0), (2, 2)]

                if not (len(self.cumulative_probs) - 1, b) in founds:
                    probs += [self.cumulative_probs[-1][b] * self.get_length_penalty(len(self.cumulative_probs),
                                                                                     alpha=length_penalty)]
                    founds += [(t, b)]

                    # cumulative_probs[-1][1] : -8.906155586242676, get_length_penalty : 0.6147386076544852

                    # probs  : [tensor(-inf), tensor(-7.3420), tensor(-7.4389), tensor(-6.3000)]
                    # founds : [(1, 1), (2, 0), (2, 2), (2, 1)]

        # print(f"probs : {probs}")       # [tensor(-inf), tensor(-7.3420), tensor(-7.4389), tensor(-6.3000)]
        # print(f"founds : {founds}")     # [(1, 1), (2, 0), (2, 2), (2, 1)]

        # Sort and take n-best.
        sorted_founds_with_probs = sorted(
            zip(founds, probs),
            key=itemgetter(1),
            reverse=True,
        )[:n]
        probs = []

        # sorted_founds_with_probs : [((2, 1), tensor(-6.3000))]
        for (end_index, b), prob in sorted_founds_with_probs:
            sentence = []

            # Trace from the end.
            for t in range(end_index, 0, -1):
                sentence = [self.word_indice[t][b]] + sentence
                b = self.beam_indice[t][b]

            sentences += [sentence]
            probs += [prob]

        # sentences : [[tensor(5), tensor(5)]] => ▁, ▁,
        # probs : [tensor(-6.3000)]

        return sentences, probs
