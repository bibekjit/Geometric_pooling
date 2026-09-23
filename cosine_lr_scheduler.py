import numpy as np


class CosineLRScheduler:
    def __init__(self, total_steps, peak_lr=1e-4, min_lr=1e-8, init_lr=1e-8, warmup_rate=0.05):
        """
        Custom lr scheduler. linear lr warm-up and cosine lr decay.
        :param total_steps: total training steps
        :param max_lr: peak lr value
        :param min_lr: lr decay value
        :param init_lr: initial lr value
        :param warmup_rate: number of warmup steps
        """
        self.min_lr = min_lr
        self.init_lr = init_lr
        self.total_setps = total_steps
        self.warmup = int(total_steps * warmup_rate)
        self.decay = total_steps - self.warmup
        self.warmup_lr = np.linspace(init_lr, peak_lr, self.warmup)
        self.max_lr = self.warmup_lr[-1]


    def __call__(self, step):

        if step <= self.warmup:

            lr = self.warmup_lr[step - 1]
        else:
            step = step - self.warmup
            lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(step * np.pi / self.decay))

        return lr