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
        # Inferred word index for each time-step. For now, initialized with initial time-step.
        self.word_indice = [torch.LongTensor(beam_size).zero_().to(self.device) + data_loader.BOS]
        # Beam index for selected word index, at each time-step.
        self.beam_indice = [torch.LongTensor(beam_size).zero_().to(self.device) - 1]
        # Cumulative log-probability for each beam.
        self.cumulative_probs = [torch.FloatTensor([.0] + [-float('inf')] * (beam_size - 1)).to(self.device)]
        # 1 if it is done else 0
        self.masks = [torch.BoolTensor(beam_size).zero_().to(self.device)]

        # We don't need to remember every time-step of hidden states:
        #       prev_hidden, prev_cell, prev_h_t_tilde
        # What we need is remember just last one.

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

    def get_batch(self):
        y_hat = self.word_indice[-1].unsqueeze(-1)
        # |y_hat| = (beam_size, 1)
        # if model != transformer:
        #     |hidden| = |cell| = (n_layers, beam_size, hidden_size)
        #     |h_t_tilde| = (beam_size, 1, hidden_size) or None
        # else:
        #     |prev_state_i| = (beam_size, length, hidden_size),
        #     where i is an index of layer.

        # print(f"get_batch - y_hat : {y_hat.shape}")
        # get_batch - y_hat : torch.Size([3, 1]) => [beam_size, 1]

        # print(f"y_hat : {y_hat}")
        # y_hat : tensor([[2], [2], [2]])       2 => <BOS>

        # prev_status : ['hidden_state', 'cell_state', 'h_t_1_tilde']
        # print(f"prev_status hidden_state : {self.prev_status['hidden_state'].shape}")
        # print(f"prev_status cell_state : {self.prev_status['cell_state'].shape}")
        # print(f"prev_status h_t_1_tilde : {self.prev_status['h_t_1_tilde'].shape if self.prev_status['h_t_1_tilde'] is not None else 'None'}")

        # hidden_state : torch.Size([4, 3, 768])    => [n_layers, beam_size, hidden_size]
        # cell_state : torch.Size([4, 3, 768])      => [n_layers, beam_size, hidden_size]
        # h_t_1_tilde : None -> torch.Size([3, 1, 768])    => [beam_size, 1, hidden_size]

        return y_hat, self.prev_status

    #@profile
    def collect_result(self, y_hat, prev_status):
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

        # print(f"masks : {self.masks}")
        # print(f"cumulative_probs : {self.cumulative_probs}")

        # masks : [tensor([False,False,False]),tensor([False,True,False])] => [current_time_step, beam_size, 1]
        # cumulative_probs : [tensor([0.,-inf,-inf]),tensor([-2.6131,-3.6800,-4.5125])] => [current_time_step, beam_size, 1]

        cumulative_prob = self.cumulative_probs[-1].masked_fill_(self.masks[-1], -float('inf'))
        cumulative_prob = y_hat + cumulative_prob.view(-1, 1, 1).expand(self.beam_size, 1, output_size)
        # |cumulative_prob| = (beam_size, 1, output_size)

        # print(f"cumulative_prob : {cumulative_prob}")
        # tensor([[[-394.6682,-402.9751,-399.8368,...,-382.3197,-447.5492,-281.3903]],
        # [[-inf,-inf,-inf,...,-inf,-inf,-inf]], [[-inf,-inf,-inf,...,-inf,-inf,-inf]]])

        # Now, we have new top log-probability and its index.
        # We picked top index as many as 'beam_size'.
        # Be aware that we picked top-k from whole batch through 'view(-1)'.

        # Following lines are using torch.topk, which is slower than torch.sort.
        # top_log_prob, top_indice = torch.topk(
        #     cumulative_prob.view(-1), # (beam_size * output_size,)
        #     self.beam_size,
        #     dim=-1,
        # )

        # Following lines are using torch.sort, instead of using torch.topk.
        top_log_prob, top_indice = cumulative_prob.view(-1).sort(descending=True)
        # print(f"top_log_prob 1 : {top_log_prob}, top_indice : {top_indice}")
        # top_log_prob : tensor([-2.6131,-3.6799,-4.5126,...,-inf,-inf,-inf]),
        # top_indice : tensor([4,3,35,...,40112,40113,40114])

        top_log_prob, top_indice = top_log_prob[:self.beam_size], top_indice[:self.beam_size]
        # print(f"top_log_prob 2 : {top_log_prob}, top_indice : {top_indice}")
        # top_log_prob: tensor([-2.6131, -3.6799, -4.5126]), top_indice: tensor([4, 3, 35])

        # |top_log_prob| = (beam_size,)
        # |top_indice| = (beam_size,)

        # Because we picked from whole batch, original word index should be calculated again.
        self.word_indice += [top_indice.fmod(output_size)]
        # print(f"word_indice : {self.word_indice}")
        # word_indice:[tensor([2,2,2]),tensor([4,3,35])] -> [tensor([2,2,2]),tensor([4,3,35]),tensor([4,3,4])]

        # Also, we can get an index of beam, which has top-k log-probability search result.
        self.beam_indice += [top_indice.div(float(output_size)).long()]
        # print(f"beam_indice : {self.beam_indice}")
        # beam_indice:[tensor([-1,-1,-1]),tensor([0,0,0])] -> [tensor([-1,-1,-1]),tensor([0,0,0]),tensor([0,0,2])]

        # Add results to history boards.
        self.cumulative_probs += [top_log_prob]
        # print(f"history cumulative_probs : {self.cumulative_probs}")
        # cumulative_probs : [tensor([0.,-inf,-inf]),tensor([-2.6131,-inf,-4.5125]),tensor([-5.2262,-inf,-7.1255])]

        self.masks += [torch.eq(self.word_indice[-1], data_loader.EOS)]     # Set finish mask if we got EOS.
        # print(f"history masks : {self.masks}")
        # masks : [tensor([False,False,False]),tensor([False,True,False]),tensor([False,True,False])]

        # Calculate a number of finished beams.
        self.done_cnt += self.masks[-1].float().sum()
        # print(f"done_cnt : {self.done_cnt}")

        # In beam search procedure, we only need to memorize latest status.
        # For seq2seq, it would be lastest hidden and cell state, and h_t_tilde.
        # The problem is hidden(or cell) state and h_t_tilde has different dimension order.
        # In other words, a dimension for batch index is different.
        # Therefore self.batch_dims stores the dimension index for batch index.
        # For transformer, lastest status is each layer's decoder output from the biginning.
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
            for b in range(self.beam_size):  # for each beam,
                if self.masks[t][b] == 1:  # if we had EOS on this time-step and beam,
                    # Take a record of penaltified log-proability.
                    probs += [self.cumulative_probs[t][b] * self.get_length_penalty(t, alpha=length_penalty)]
                    founds += [(t, b)]

                    # print(f"cumulative_probs[{t}][{b}] : {self.cumulative_probs[t][b]}")
                    # print(f"get_length_penalty({t})    : {self.get_length_penalty(t, alpha=length_penalty)}")
                    # print(f"probs : {probs}")       # [tensor(-inf),tensor(-inf),tensor(-6.3061)]
                    # print(f"founds : {founds}")     # [(1,1),(2,1),(3,1)]

        # Also, collect log-probability from last time-step, for the case of EOS is not shown.
        for b in range(self.beam_size):
            if self.cumulative_probs[-1][b] != -float('inf'):   # If this beam does not have EOS,
                # print(f"cumulative_probs[-1][{b}] : {self.cumulative_probs[-1][b]}")

                if not (len(self.cumulative_probs) - 1, b) in founds:
                    probs += [self.cumulative_probs[-1][b] * self.get_length_penalty(len(self.cumulative_probs),
                                                                                     alpha=length_penalty)]
                    founds += [(t, b)]

                    # print(f"get_length_penalty : {self.get_length_penalty(len(self.cumulative_probs),alpha=length_penalty)}")
                    # cumulative_probs[-1][1] : -8.906155586242676, get_length_penalty : 0.6147386076544852
                    # print(f"probs : {probs}")       # [tensor(-inf),tensor(-inf),tensor(-6.3061),tensor(-4.8190),tensor(-5.9867)]
                    # print(f"founds : {founds}")     # [(1,1),(2,1),(3,1),(3,0),(3,2)]

        # Sort and take n-best.
        sorted_founds_with_probs = sorted(
            zip(founds, probs),
            key=itemgetter(1),
            reverse=True,
        )[:n]
        probs = []

        for (end_index, b), prob in sorted_founds_with_probs:
            sentence = []

            # Trace from the end.
            for t in range(end_index, 0, -1):
                sentence = [self.word_indice[t][b]] + sentence
                b = self.beam_indice[t][b]

            sentences += [sentence]
            probs += [prob]

        print(f"sentences : {sentences}")
        print(f"probs : {probs}")

        # sentences : [[tensor(4),tensor(4),tensor(4)]] => ...
        # probs : [tensor(-4.8190)]

        return sentences, probs
