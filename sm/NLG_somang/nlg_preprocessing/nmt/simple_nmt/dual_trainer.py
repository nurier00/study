import numpy as np

import torch
from torch import optim
import torch.nn.utils as torch_utils
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler

from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar

from simple_nmt.utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2

X2Y, Y2X = 0, 1


class DualSupervisedTrainingEngine(Engine):

    def __init__(
            self,
            func,
            models,
            crits,
            optimizers,
            lr_schedulers,
            language_models,
            config
    ):
        self.models = models
        self.crits = crits
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers
        self.language_models = language_models
        self.config = config

        super().__init__(func)

        self.best_x2y = np.inf
        self.best_y2x = np.inf
        self.scalers = [
            GradScaler(),
            GradScaler(),
        ]

    @staticmethod
    def _reorder(x, y, l):
        # This method is one of important methods in this class.
        # Since encoder takes packed_sequence instance,
        # the samples in mini-batch must be sorted by lengths.
        # Thus, we need to re-order the samples in mini-batch, if src and tgt is reversed.
        # (Because originally src and tgt are sorted by the length of samples in src.)

        # packed sequence 를 사용하기 때문에 sample 미니배치는 늘 길이가 긴 것부터 정렬되어 있어야 함
        # 따라서 x 에 맞추어 정렬이 되어 있었던 배치를 y 의 미니배치가 길이가 긴 것부터 정렬 되도록 y 에 맞추어 재정렬 해주어야함

        # sort by length.
        indice = l.sort(descending=True)[1]

        # re-order based on the indice.
        x_ = x.index_select(dim=0, index=indice).contiguous()
        y_ = y.index_select(dim=0, index=indice).contiguous()
        l_ = l.index_select(dim=0, index=indice).contiguous()

        # x_ shape : torch.Size([64, 43]) => [bs, length]
        # 각 문장의 vocab 인덱스로 구성되어진 텐서
        '''
        x_ : tensor([[   2,   18,  525,  ..., 5194,    6,    3],
        [   2,   42,    5,  ...,    6,    3,    1],
        [   2,   18,  325,  ..., 2982,    6,    3],
        ...,
        [   2,   51, 1126,  ...,    6,   17,    3],
        [   2,   18,  701,  ..., 1510,    6,    3],
        [   2,   97,  329,  ..., 5046,    6,    3]])
        '''

        # y_ shape : torch.Size([64, 60]) => [bs, length]
        '''
        y_ : tensor([[ 460,   40, 1093,  ...,  541,   30,    4],
        [ 228,  201,   40,  ...,    1,    1,    1],
        [ 732,  573, 2919,  ...,    1,    1,    1],
        ...,
        [1497,  815,  191,  ...,    1,    1,    1],
        [ 913,  443, 2242,  ...,    1,    1,    1],
        [ 503, 3259,  516,  ...,    1,    1,    1]])
        '''

        # l_ : torch.Size([64]) => [bs]
        '''
        l_ : tensor([60, 56, 53, 52, 52, 49, 48, 47, 46, 46, 45, 44, 44, 44, 43, 43, 43, 43,
        42, 42, 42, 41, 41, 41, 41, 40, 40, 40, 39, 39, 39, 39, 39, 38, 38, 38,
        38, 38, 37, 37, 37, 37, 37, 37, 37, 37, 37, 36, 36, 35, 35, 35, 35, 34,
        34, 34, 34, 34, 34, 34, 32, 32, 32, 30])
        '''

        # generate information to restore the re-ordering.
        restore_indice = indice.sort(descending=False)[1]
        # restore_indice shape : torch.Size([64]) => [bs]
        '''
        restore_indice : tensor([27, 30, 29, 34, 31, 33, 41, 40, 45, 46, 49, 53, 58, 60,  0,  4,  3,  7,
         8, 14, 13, 12, 17, 21, 24, 23, 22, 25, 26, 36, 35, 38, 47, 50, 51, 48,
        52, 54, 55, 56, 57, 59, 61, 62, 63,  1,  2,  5,  6, 10,  9, 16, 15, 11,
        20, 18, 19, 28, 37, 32, 43, 42, 39, 44])
        '''

        return x_, (y_, l_), restore_indice

    @staticmethod
    def _restore_order(x, restore_indice):
        # reorder 시켜주었던 배치를 원래대로 x 에 맞추어 정렬 원상복구 시켜줌
        return x.index_select(dim=0, index=restore_indice)

    @staticmethod
    def _get_loss(x, y, x_hat, y_hat, crits, x_lm=None, y_lm=None, lagrange=1e-3):
        # 실제 정답의 원핫인덱스가 들어있는 텐서
        # |x| = (batch_size, n)
        # |y| = (batch_size, m)

        # seq2seq 로 부터 나온 log likely-hood, 즉, 로그확률분포
        # |x_hat| = (batch_size, n, output_size0) -> log P (x|y)
        # |y_hat| = (batch_size, m, output_size1) -> log P (y|x)

        # log P x 를 위한 샘플별 스텝별 로그 확률값
        # |x_lm| = |x_hat|
        # log P y 를 위한 샘플별 스텝별 로그 확률값
        # |y_lm| = |y_hat|

        log_p_y_given_x = -crits[X2Y](
            y_hat.contiguous().view(-1, y_hat.size(-1)),
            y.contiguous().view(-1),
        )
        # y_hat.view() : torch.Size([3835, 6249])   => [bs * m, output_size1]
        # y.view() : torch.Size([3835])             => [bs * m]
        # log_p_y_given_x shape : torch.Size([3835])

        log_p_x_given_y = -crits[Y2X](
            x_hat.contiguous().view(-1, x_hat.size(-1)),
            x.contiguous().view(-1),
        )
        # x_hat.view() : torch.Size([3835, 8101])   => [bs * n, output_size0]
        # x.view() : torch.Size([3835])             => [bs * n]
        # log_p_x_given_y shape : torch.Size([3835])

        # |log_p_y_given_x| = (batch_size * m)
        # |log_p_x_given_y| = (batch_size * n)

        log_p_y_given_x = log_p_y_given_x.view(y.size(0), -1).sum(dim=-1)
        log_p_x_given_y = log_p_x_given_y.view(x.size(0), -1).sum(dim=-1)
        # |log_p_y_given_x| = |log_p_x_given_y| = (batch_size, )

        # Negative Log-likelihood
        loss_x2y = -log_p_y_given_x
        loss_y2x = -log_p_x_given_y

        if x_lm is not None and y_lm is not None:
            log_p_x = -crits[Y2X](
                x_lm.contiguous().view(-1, x_lm.size(-1)),
                x.contiguous().view(-1),
            )
            log_p_y = -crits[X2Y](
                y_lm.contiguous().view(-1, y_lm.size(-1)),
                y.contiguous().view(-1),
            )
            # |log_p_x| = (batch_size * n)
            # |log_p_y| = (batch_size * m)

            log_p_x = log_p_x.view(x.size(0), -1).sum(dim=-1)
            log_p_y = log_p_y.view(y.size(0), -1).sum(dim=-1)
            # |log_p_x| = (batch_size, )
            # |log_p_y| = (batch_size, )

            # Just for logging: both losses are detached.
            dual_loss = lagrange * ((log_p_x + log_p_y_given_x.detach()) - (log_p_y + log_p_x_given_y.detach())) ** 2

            # Note that 'detach()' is used to prevent unnecessary back-propagation.
            loss_x2y += lagrange * ((log_p_x + log_p_y_given_x) - (log_p_y + log_p_x_given_y.detach())) ** 2
            loss_y2x += lagrange * ((log_p_x + log_p_y_given_x.detach()) - (log_p_y + log_p_x_given_y)) ** 2
        else:
            dual_loss = None

        return (
            loss_x2y.sum(),
            loss_y2x.sum(),
            float(dual_loss.sum()) if dual_loss is not None else .0,
        )

    @staticmethod
    def train(engine, mini_batch):
        for language_model, model, optimizer in zip(engine.language_models,
                                                    engine.models,
                                                    engine.optimizers):
            # language_model 은 이미 pre-training 한 상태, 다시 train X
            language_model.eval()
            model.train()
            if engine.state.iteration % engine.config.iteration_per_update == 1 or \
                    engine.config.iteration_per_update == 1:
                if engine.state.iteration > 1:
                    optimizer.zero_grad()

        device = next(engine.models[0].parameters()).device
        mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1].to(device))
        mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1].to(device))

        with autocast(not engine.config.off_autocast):
            # X2Y => data loader 에서 넘어온 형태, 구현이 짧음
            # x 는 Encoder 에 들어갈 값이므로 <BOS>, <EOS> 를 제거해줌, Decoder 에 들어갈 y 는 <EOS> 를 제거
            x, y = (mini_batch.src[0][:, 1:-1], mini_batch.src[1] - 2), mini_batch.tgt[0][:, :-1]
            x_hat_lm, y_hat_lm = None, None
            # |x| = (batch_size, n)
            # |y| = (batch_size, m)

            # teacher Forcing
            y_hat = engine.models[X2Y](x, y)
            # |y_hat| = (batch_size, m, y_vocab_size)
            # y_hat 에는 각 미니배치의 샘플별 각 타임스텝별 각 단어별 확률값이 들어있음

            if engine.state.epoch > engine.config.dsl_n_warmup_epochs:  # warmup 상태가 지났는지 확인
                with torch.no_grad():  # warmup 상태가 끝이나면 gradient 계산 X
                    y_hat_lm = engine.language_models[X2Y](y)  # log P (y)
                    # |y_hat_lm| = |y_hat| = (batch_size, m, y_vocab_size)

            # Y2X
            # Since encoder in seq2seq takes packed_sequence instance,
            # we need to re-sort if we use reversed src and tgt.

            # 정렬이 X2Y 기준으로 되어있기 때문에 재정렬 필요
            x, y, restore_indice = DualSupervisedTrainingEngine._reorder(
                mini_batch.src[0][:, :-1],  # => Decoder 에 들어가야 하기 때문에 <EOS> 를 제거
                mini_batch.tgt[0][:, 1:-1],  # => Encoder 에 들어가야 하기 때문에 <BOS>, <EOS> 를 제거
                mini_batch.tgt[1] - 2,
            )
            # |x| = (batch_size, n)
            # |y| = (batch_size, m)

            x_hat = DualSupervisedTrainingEngine._restore_order(
                engine.models[Y2X](y, x),
                restore_indice=restore_indice,
            )
            # |x_hat| = (batch_size, n, x_vocab_size)

            if engine.state.epoch > engine.config.dsl_n_warmup_epochs:
                with torch.no_grad():
                    x_hat_lm = DualSupervisedTrainingEngine._restore_order(
                        engine.language_models[Y2X](x),
                        restore_indice=restore_indice,
                    )
                    # |x_hat_lm| = |x_hat|

            # Decoder 의 정답으로 쓰일 x, y 이기 때문에 둘 다 <BOS> 를 제거
            x, y = mini_batch.src[0][:, 1:], mini_batch.tgt[0][:, 1:]
            loss_x2y, loss_y2x, dual_loss = DualSupervisedTrainingEngine._get_loss(
                x, y,
                x_hat, y_hat,
                engine.crits,
                x_hat_lm, y_hat_lm,
                # According to the paper, DSL should be warm-started.
                # Thus, we turn-off the regularization at the beginning.
                lagrange=engine.config.dsl_lambda if engine.state.epoch > engine.config.dsl_n_warmup_epochs else .0
                # 라그랑쥬는 warmup 단계에서는 0 warmup 단계가 끝난경우 실제 lambda 값을 넣어줌
            )

            backward_targets = [
                loss_x2y.div(y.size(0)).div(engine.config.iteration_per_update),
                loss_y2x.div(x.size(0)).div(engine.config.iteration_per_update),
            ]

        for scaler, backward_target in zip(engine.scalers, backward_targets):
            if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
                scaler.scale(backward_target).backward()
            else:
                backward_target.backward()

        x_word_count = int(mini_batch.src[1].sum())
        y_word_count = int(mini_batch.tgt[1].sum())
        p_norm = float(get_parameter_norm(list(engine.models[X2Y].parameters()) +
                                          list(engine.models[Y2X].parameters())))
        g_norm = float(get_grad_norm(list(engine.models[X2Y].parameters()) +
                                     list(engine.models[Y2X].parameters())))

        if engine.state.iteration % engine.config.iteration_per_update == 0 and \
                engine.state.iteration > 0:
            for model, optimizer, scaler in zip(engine.models,
                                                engine.optimizers,
                                                engine.scalers):
                torch_utils.clip_grad_norm_(
                    model.parameters(),
                    engine.config.max_grad_norm,
                )
                # Take a step of gradient descent.
                if engine.config.gpu_id >= 0 and not engine.config.off_autocast:
                    # Use scaler instead of engine.optimizer.step()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

        return {
            'x2y': float(loss_x2y / y_word_count),
            'y2x': float(loss_y2x / x_word_count),
            'reg': float(dual_loss / x.size(0)),
            '|param|': p_norm if not np.isnan(p_norm) and not np.isinf(p_norm) else 0.,
            '|g_param|': g_norm if not np.isnan(g_norm) and not np.isinf(g_norm) else 0.,
        }

    @staticmethod
    def validate(engine, mini_batch):
        for model in engine.models:
            model.eval()

        with torch.no_grad():
            device = next(engine.models[0].parameters()).device
            mini_batch.src = (mini_batch.src[0].to(device), mini_batch.src[1].to(device))
            mini_batch.tgt = (mini_batch.tgt[0].to(device), mini_batch.tgt[1].to(device))

            with autocast(not engine.config.off_autocast):
                # X2Y
                x, y = (mini_batch.src[0][:, 1:-1], mini_batch.src[1] - 2), mini_batch.tgt[0][:, :-1]
                # |x| = (batch_size, n)
                # |y| = (batch_size  m)
                y_hat = engine.models[X2Y](x, y)
                # |y_hat| = (batch_size, m, y_vocab_size)

                # Y2X
                x, y, restore_indice = DualSupervisedTrainingEngine._reorder(
                    mini_batch.src[0][:, :-1],
                    mini_batch.tgt[0][:, 1:-1],
                    mini_batch.tgt[1] - 2,
                )
                x_hat = DualSupervisedTrainingEngine._restore_order(
                    engine.models[Y2X](y, x),
                    restore_indice=restore_indice,
                )
                # |x_hat| = (batch_size, n, x_vocab_size)

                # You don't have to use _get_loss method,
                # because we don't have to care about the gradients.
                x, y = mini_batch.src[0][:, 1:], mini_batch.tgt[0][:, 1:]
                loss_x2y = engine.crits[X2Y](
                    y_hat.contiguous().view(-1, y_hat.size(-1)),
                    y.contiguous().view(-1)
                ).sum()
                loss_y2x = engine.crits[Y2X](
                    x_hat.contiguous().view(-1, x_hat.size(-1)),
                    x.contiguous().view(-1)
                ).sum()

                x_word_count = int(mini_batch.src[1].sum())
                y_word_count = int(mini_batch.tgt[1].sum())

        return {
            'x2y': float(loss_x2y / y_word_count),
            'y2x': float(loss_y2x / x_word_count),
        }

    @staticmethod
    def attach(
            train_engine,
            validation_engine,
            training_metric_names=['x2y', 'y2x', 'reg', '|param|', '|g_param|'],
            validation_metric_names=['x2y', 'y2x'],
            verbose=VERBOSE_BATCH_WISE
    ):
        # Attaching would be repaeted for serveral metrics.
        # Thus, we can reduce the repeated codes by using this function.
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name,
            )

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_logs(engine):
                avg_p_norm = engine.state.metrics['|param|']
                avg_g_norm = engine.state.metrics['|g_param|']
                avg_x2y = engine.state.metrics['x2y']
                avg_y2x = engine.state.metrics['y2x']
                avg_reg = engine.state.metrics['reg']

                print(
                    'Epoch {} - |param|={:.2e} |g_param|={:.2e} loss_x2y={:.4e} ppl_x2y={:.2f} loss_y2x={:.4e} ppl_y2x={:.2f} dual_loss={:.4e}'.format(
                        engine.state.epoch,
                        avg_p_norm,
                        avg_g_norm,
                        avg_x2y, np.exp(avg_x2y),
                        avg_y2x, np.exp(avg_y2x),
                        avg_reg,
                    ))

        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                avg_x2y = engine.state.metrics['x2y']
                avg_y2x = engine.state.metrics['y2x']

                print('Validation X2Y - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}'.format(
                    avg_x2y,
                    np.exp(avg_x2y),
                    engine.best_x2y,
                    np.exp(engine.best_x2y),
                ))
                print('Validation Y2X - loss={:.4e} ppl={:.2f} best_loss={:.4e} best_ppl={:.2f}'.format(
                    avg_y2x,
                    np.exp(avg_y2x),
                    engine.best_y2x,
                    np.exp(engine.best_y2x),
                ))

    @staticmethod
    def resume_training(engine, resume_epoch):
        engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
        engine.state.epoch = (resume_epoch - 1)

    @staticmethod
    def check_best(engine):
        x2y = float(engine.state.metrics['x2y'])
        if x2y <= engine.best_x2y:
            engine.best_x2y = x2y
        y2x = float(engine.state.metrics['y2x'])
        if y2x <= engine.best_y2x:
            engine.best_y2x = y2x

    @staticmethod
    def save_model(engine, train_engine, config, vocabs):
        avg_train_x2y = train_engine.state.metrics['x2y']
        avg_train_y2x = train_engine.state.metrics['y2x']
        avg_valid_x2y = engine.state.metrics['x2y']
        avg_valid_y2x = engine.state.metrics['y2x']

        # Set a filename for model of last epoch.
        # We need to put every information to filename, as much as possible.
        model_fn = config.model_fn.split('.')

        model_fn = model_fn[:-1] + ['%02d' % train_engine.state.epoch,
                                    '%.2f-%.2f' % (avg_train_x2y,
                                                   np.exp(avg_train_x2y)
                                                   ),
                                    '%.2f-%.2f' % (avg_train_y2x,
                                                   np.exp(avg_train_y2x)
                                                   ),
                                    '%.2f-%.2f' % (avg_valid_x2y,
                                                   np.exp(avg_valid_x2y)
                                                   ),
                                    '%.2f-%.2f' % (avg_valid_y2x,
                                                   np.exp(avg_valid_y2x)
                                                   ),
                                    ] + [model_fn[-1]]

        model_fn = '.'.join(model_fn)

        torch.save(
            {
                'model': [
                    train_engine.models[0].state_dict(),
                    train_engine.models[1].state_dict(),
                    train_engine.language_models[0].state_dict(),
                    train_engine.language_models[1].state_dict(),
                ],
                'opt': [
                    train_engine.optimizers[0].state_dict(),
                    train_engine.optimizers[1].state_dict(),
                ],
                'config': config,
                'src_vocab': vocabs[0],
                'tgt_vocab': vocabs[1],
            }, model_fn
        )


class DualSupervisedTrainer():

    def __init__(self, config):
        self.config = config

    def train(
            self,
            models, language_models,
            crits, optimizers,
            train_loader, valid_loader,
            vocabs,
            n_epochs,
            lr_schedulers=None
    ):
        # Declare train and validation engine with necessary objects.
        train_engine = DualSupervisedTrainingEngine(
            DualSupervisedTrainingEngine.train,
            models,
            crits,
            optimizers,
            lr_schedulers,
            language_models,
            self.config,
        )
        validation_engine = DualSupervisedTrainingEngine(
            DualSupervisedTrainingEngine.validate,
            models,
            crits,
            optimizers=None,
            lr_schedulers=None,
            language_models=language_models,
            config=self.config,
        )

        # Do necessary attach procedure to train & validation engine.
        # Progress bar and metric would be attached.
        DualSupervisedTrainingEngine.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        # After every train epoch, run 1 validation epoch.
        # Also, apply LR scheduler if it is necessary.
        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

            if engine.lr_schedulers is not None:
                for s in engine.lr_schedulers:
                    s.step()

        # Attach above call-back function.
        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            run_validation,
            validation_engine,
            valid_loader
        )
        # Attach other call-back function for initiation of the training.
        train_engine.add_event_handler(
            Events.STARTED,
            DualSupervisedTrainingEngine.resume_training,
            self.config.init_epoch,
        )

        # Attach validation loss check procedure for every end of validation epoch.
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED, DualSupervisedTrainingEngine.check_best
        )
        # Attach model save procedure for every end of validation epoch.
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            DualSupervisedTrainingEngine.save_model,
            train_engine,
            self.config,
            vocabs,
        )

        # Start training.
        train_engine.run(train_loader, max_epochs=n_epochs)

        return models
