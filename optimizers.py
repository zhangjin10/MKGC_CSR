"""
Optimizers and learning rate schedulers
"""

import torch
from torch.optim import Optimizer
import math


class WarmupScheduler:
    """Learning rate warmup scheduler"""

    def __init__(self, optimizer, warmup_steps, initial_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.current_step = 0

        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        """Update learning rate"""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            lr_scale = self.current_step / self.warmup_steps

            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * lr_scale


class CosineAnnealingWarmup:
    """Cosine annealing with warmup"""

    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.current_step = 0

        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        """Update learning rate"""
        self.current_step += 1

        if self.current_step <= self.warmup_steps:
            lr_scale = self.current_step / self.warmup_steps

            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * lr_scale

        else:
            progress = (self.current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))

            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.min_lr + (self.base_lrs[i] - self.min_lr) * cosine_decay


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay"""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Perform optimization step"""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                step_size = group['lr'] / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                p.data.add_(p.data, alpha=-group['lr'] * group['weight_decay'])

        return loss


class Lookahead(Optimizer):
    """Lookahead optimizer wrapper"""

    def __init__(self, base_optimizer, k=5, alpha=0.5):
        self.base_optimizer = base_optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.base_optimizer.param_groups

        self.slow_weights = [[p.clone().detach() for p in group['params']]
                            for group in self.param_groups]

        self.step_counter = 0

    def step(self, closure=None):
        """Perform optimization step"""
        loss = self.base_optimizer.step(closure)
        self.step_counter += 1

        if self.step_counter % self.k == 0:
            for group, slow_weight_group in zip(self.param_groups, self.slow_weights):
                for p, slow_weight in zip(group['params'], slow_weight_group):
                    slow_weight.data.add_(p.data - slow_weight.data, alpha=self.alpha)
                    p.data.copy_(slow_weight.data)

        return loss

    def zero_grad(self):
        """Zero gradients"""
        self.base_optimizer.zero_grad()


class GradientClipping:
    """Gradient clipping utility"""

    def __init__(self, max_norm=1.0, norm_type=2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip(self, parameters):
        """Clip gradients"""
        return torch.nn.utils.clip_grad_norm_(parameters, self.max_norm, self.norm_type)


class EMA:
    """Exponential Moving Average of model parameters"""

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data

    def apply_shadow(self):
        """Apply shadow parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
