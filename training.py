import math
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


class ConstantLRWithWarmup(LambdaLR):
    """
    Increase lr to optimizer.lr linearly with ratio cur_step/num_warmup_steps
    If cur_step >= num_warmup_steps, the ratio will be 1.0
    Example:
        >>> scheduler = ConstantLRWithWarmup(optimizer, num_warmup_steps, last_epoch)
        ...
        >>> loss.backward()
        >>> optimizer.step()
        >>> scheduler.step()

    """
    def __init__(self, optimizer, num_warmup_steps, last_epoch=-1):
        def lr_lambda(cur_step):
            if cur_step < num_warmup_steps:
                return float(cur_step) / float(max(num_warmup_steps, 1.0))
            return 1.0
        super(ConstantLRWithWarmup, self).__init__(optimizer, lr_lambda, last_epoch)


class ExpDecayLRWithWarmup(LambdaLR):
    """
    Increase lr to optimizer.lr linearly with ratio cur_step/num_warmup_steps
    If cur_step >= num_warmup_steps, the optimizer.lr will be multiplied by decay_factor for each step
    Example:
        >>> scheduler = ExpDecayLRWithWarmup(optimizer, num_warmup_steps, decay_factor, last_epoch)
        ...
        >>> loss.backward()
        >>> optimizer.step()
        >>> scheduler.step()
    """
    def __init__(self, optimizer, num_warmup_steps, decay_factor=0.9995, last_epoch=-1):
        def lr_lambda(cur_step):
            if cur_step < num_warmup_steps:
                return float(cur_step) / float(max(num_warmup_steps, 1.0))
            times = min(max(cur_step, 1.0) - num_warmup_steps, 1.0)
            return decay_factor**times
        super(ExpDecayLRWithWarmup, self).__init__(optimizer, lr_lambda, last_epoch)


class GradientAccumulator(object):
    def __init__(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


if __name__ == "__main__":
    t = torch.Tensor([5,6])
    t.requires_grad_()
    opt = torch.optim.AdamW([t], lr=1e-3)
    a = ConstantLRWithWarmup(opt, 1000)
    a.step()
    print(opt.param_groups[0]["lr"]) # 1e-6
    a.step()
    print(opt.param_groups[0]["lr"]) # 1e-6