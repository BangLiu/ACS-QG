import math
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer


class MyAdam(Optimizer):
    """Implements Adam algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(MyAdam, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = grad.new().resize_as_(grad).zero_()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = grad.new().resize_as_(grad).zero_()

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = exp_avg_sq.sqrt().add_(
                    group['eps'] * math.sqrt(bias_correction2))
                step_size = group['lr'] * math.sqrt(
                    bias_correction2) / bias_correction1

                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss


class Optim(object):
    def set_parameters(self, params):
        self.params = list(params)  # careful: params may be a generator
        if self.method == 'sgd':
            self.optimizer = optim.SGD(self.params, lr=self.lr)
        elif self.method == 'adagrad':
            self.optimizer = optim.Adagrad(self.params, lr=self.lr)
        elif self.method == 'adadelta':
            self.optimizer = optim.Adadelta(self.params, lr=self.lr)
        elif self.method == 'adam':
            # self.optimizer = optim.Adam(self.params, lr=self.lr)
            self.optimizer = MyAdam(self.params, lr=self.lr)
        else:
            raise RuntimeError("Invalid optim method: " + self.method)

    def __init__(self, method, lr, max_grad_norm, max_weight_value=None,
                 lr_decay=1, start_decay_at=None, decay_bad_count=6):
        self.last_ppl = None
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.max_weight_value = max_weight_value
        self.method = method
        self.lr_decay = lr_decay
        self.start_decay_at = start_decay_at
        self.start_decay = False
        self.decay_bad_count = decay_bad_count
        self.best_metric = 0
        self.bad_count = 0

    def step(self):
        # Compute gradients norm.
        if self.max_grad_norm:
            clip_grad_norm_(self.params, self.max_grad_norm)
        self.optimizer.step()
        if self.max_weight_value:
            for p in self.params:
                p.data.clamp_(0 - self.max_weight_value, self.max_weight_value)

    def updateLearningRate(self, ppl, epoch):
        """
        Decay learning rate if val perf does not improve or
        we hit the start_decay_at limit
        """
        if ppl >= self.best_metric:
            self.best_metric = ppl
            self.bad_count = 0
        else:
            self.bad_count += 1
        print('Bad_count: {0}\tCurrent lr: {1}'.format(
            self.bad_count, self.lr))
        print('Best metric: {0}'.format(
            self.best_metric))

        if self.bad_count >= self.decay_bad_count and self.lr >= 1e-6:
            self.lr = self.lr * self.lr_decay
            print("Decaying learning rate to %g" % self.lr)
            self.bad_count = 0
        self.optimizer.param_groups[0]['lr'] = self.lr
