import math
import numpy as np
import torch
from torch import nn
from torch.functional import Tensor
from torch.nn.modules.loss import CrossEntropyLoss
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
        optimizer.zero_grad()
        optimizer.step()


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
        optimizer.zero_grad()
        optimizer.step()


class GradientAccumulator(object):
    """
    Update parameter after accumulate gradient by given accumulation_steps \n
    Example:
        >>> accumulator = GradientAccumulator(accumulation_steps=3)
        >>> optimizer = torch.optim.Adam(model.parameters())
        ...
        >>> loss = loss_func(predictions, target)
        >>> with accumulator(loss, optimizer) as accu:
                accu.backward()
                accu.step()
    """
    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self._cur_step = 0

    def __call__(self, loss, optimizer, lr_scheduler=None):
        self._loss = loss
        self._optimizer = optimizer
        self._lr_scheduler = lr_scheduler
        return self

    def __enter__(self):
        self._cur_step += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
    
    def backward(self):
        self._loss = self._loss / self.accumulation_steps
        self._loss.backward()
    
    def step(self):
        if self._cur_step % self.accumulation_steps == 0:
            # change lr before apply gradients to weights
            if self._lr_scheduler is not None:
                self._lr_scheduler.step()
            self._optimizer.step()
            self._optimizer.zero_grad()
    
    @property
    def cur_step(self):
        return self._cur_step
    



if __name__ == "__main__":
    def test_accumulator():
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        from torch.utils.data import TensorDataset, DataLoader
        data = load_iris()
        X = data['data']
        y = data['target']
        model = torch.nn.Sequential(nn.Linear(4,16), nn.Linear(16,3))
        opti = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = CrossEntropyLoss()
        accumulator = GradientAccumulator(accumulation_steps=3)
        
        tx, cvx, ty, cvy = train_test_split(X, y, test_size=0.2, random_state=1)
        train_loader = DataLoader(TensorDataset(torch.FloatTensor(tx), torch.LongTensor(ty)), batch_size=32)
        dev_loader = DataLoader(TensorDataset(torch.FloatTensor(cvx), torch.LongTensor(cvy)), batch_size=32)
        for i in range(100):
            for train_x, train_y in train_loader:
                output = model(train_x)
                loss = criterion(output, train_y)
                with accumulator(loss, opti) as accu:
                    accu.backward()
                    accu.step()
            for dev_x, dev_y in train_loader:
                output = model(dev_x)
                true_num = (output.argmax(dim=1) == dev_y).sum()
            print("Correct num: {}".format(true_num.item()))
    test_accumulator()

    