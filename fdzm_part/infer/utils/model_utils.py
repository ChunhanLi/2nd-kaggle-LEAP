import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup, \
  get_cosine_schedule_with_warmup, \
  get_polynomial_decay_schedule_with_warmup, get_constant_schedule_with_warmup


def get_scheduler(optimizer, CFG, num_train_steps):

    if CFG.scheduler_type == 'constant_schedule_with_warmup':
        scheduler = get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=CFG.n_warmup_steps
        )
    elif CFG.scheduler_type == 'linear_schedule_with_warmup':
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=CFG.n_warmup_steps,
            num_training_steps=num_train_steps
        )
    elif CFG.scheduler_type == 'cosine_schedule_with_warmup':
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=CFG.n_warmup_steps,
            num_cycles=CFG.cosine_n_cycles, # 代表num_train_steps-num_warmup_steps中有几个cos周期 一个pai算一个
            num_training_steps=num_train_steps, # 一个周期多长 要加上warmup的 以num_train_steps-num_warmup_steps算
        )
#     elif CFG.scheduler_type == 'polynomial_decay_schedule_with_warmup':
#         scheduler = get_polynomial_decay_schedule_with_warmup(
#             optimizer,
#             num_warmup_steps=CFG.n_warmup_steps,
#             num_training_steps=num_train_steps,
#             power=config.scheduler.polynomial_decay_schedule_with_warmup.power,
#             lr_end=config.scheduler.polynomial_decay_schedule_with_warmup.min_lr
#         )
    else:
        raise ValueError(f'Unknown scheduler: {CFG.scheduler.scheduler_type}')

    return scheduler



class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# # 初始化
# ema = EMA(model, 0.999)
# ema.register()

# # 训练过程中，更新完参数后，同步update shadow weights
# def train():
#     optimizer.step()
#     ema.update()

# # eval前，apply shadow weights；eval之后，恢复原来模型的参数
# def evaluate():
#     ema.apply_shadow()
#     # evaluate
#     ema.restore()