import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import json
import gc
import warnings
from warnings import simplefilter
from sklearn.metrics import r2_score
from box import Box
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import _LRScheduler
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup, get_constant_schedule_with_warmup
import time
from tqdm.auto import tqdm
import math
import sys
sys.path.insert(0, "./")
import os
import gc
import functools
from glob import glob

from utils.model_utils import *
from utils.base_utils import *

exp_id = "ex912sep"
os.makedirs(f"../outputs/{exp_id}", exist_ok=True)

config = {
    # ======================= global setting =====================
    "debug": False,
    "exp_id": exp_id,
    "seed": 315,
    "log_path": f"../outputs/{exp_id}/log.txt",
    "save_path": f"../outputs/{exp_id}",
    "device": "cuda:0",
    "save_best_model": True,
    # ======================== train & val ==========================
    "epoch": 1,
    "print_freq": 10000,
    "gradient_checkpointing_enable": False,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 4,
    "evaluate_n_times_per_epoch": 2,
    "early_stop_flag": True,
    "early_stop_step": 1000, # 1的话其实已经是两个epoch不增长了
    "batch_size": 1024*3,
    "val_batch_size": 2048*3,
    "eval_val": True,
    "num_workers": 12,
    "apex": False,
    "scheduler_type": "CosineAnnealingWarmRestarts",  # cosine_schedule_with_warmup, plateau, CosineAnnealingWarmRestarts
    "n_warmup_steps": 0,
    "cosine_n_cycles": 0.5,
    "learning_rate": 8e-4,
    "lr_factor": 0.2,
    "lr_patience": 3,
    "dropout_rate": 0.2,
    "lr_lstm": 6e-4,
    "lr_gru": 6e-4,
    "lr_attention": 4e-4,
    "lr_fc": 8e-4
}

if config["debug"]:
    config["epoch"] = 1
    config["print_freq"] = 50

CFG = Box(config)
logger = get_logger(filename=CFG.log_path, filemode='w', add_console=False)
seed_everything(seed=CFG.seed)


input_single_num = 16
input_series_num = 9
output_single_num = 8
output_series_num = 6

logger.info(f"exp id: {exp_id}")
logger.info(f"config: {config}")

ptend_index = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67]
ptend_q0001_index = [68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127]
ptend_q0002_index = [128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187]
ptend_q0003_index = [188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247]
ptend_u_index = [248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307]
ptend_v_index = [308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367]


class LEAPLoss(nn.Module):
    def __init__(self, w=1):
        super(LEAPLoss, self).__init__()
        self.l1_loss = nn.SmoothL1Loss()
        self.w = w

    def forward(self, y_pred, y_true, gid=None):
        if gid is None:
            return torch.FloatTensor([-1])
        else:
            idx = GIDX_DICT[gid] # GIDX_LOSS_DICT
            loss = self.l1_loss(y_pred[:, idx], y_true[:, idx])
            if gid > 0:
                i = gid - 1
                pred_diff = y_pred[:, 8+60*i+1:8+60*(i+1)] - y_pred[:, 8+60*i:8+60*(i+1)-1]
                label_diff = y_true[:, 8+60*i+1:8+60*(i+1)] - y_true[:, 8+60*i:8+60*(i+1)-1]
                loss += self.l1_loss(pred_diff, label_diff)*self.w
        
            return loss


class Multisample_Dropout(nn.Module):
    def __init__(self):
        super(Multisample_Dropout, self).__init__()
        self.dropout = nn.Dropout(.1)
        self.dropouts = nn.ModuleList([nn.Dropout((i+1)*.1) for i in range(4)])
        
    def forward(self, x, module):
        x = self.dropout(x)
        return torch.mean(torch.stack([module(dropout(x)) for dropout in self.dropouts], dim=0), dim=0)


def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class TimeDistributed(nn.Module):
    def __init__(self, module: nn.Module, batch_first: bool = True):
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)
        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)
        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)
        return y


class CNNLSTMBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.conv = nn.Conv1d(self.input_size, self.hidden_size, kernel_size=5, padding='same')
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, num_layers=1, bidirectional=True, batch_first=True)
        self.timefc = TimeDistributed(nn.Linear(self.hidden_size*2, self.hidden_size), batch_first=True)
        self.resfc = TimeDistributed(nn.Linear(self.input_size, self.hidden_size), batch_first=True)

    def forward(self, x):
        if self.input_size != self.hidden_size:
            x_res = self.resfc(x)
        else:
            x_res = x
        x = self.conv(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.timefc(x)
        x = x + x_res
        x = nn.ReLU()(x)

        return x


class LEAP_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.lstm_layers = nn.ModuleList()
        self.lstm_layers.append(CNNLSTMBlock(25, 256))
        for i in range(7):
            self.lstm_layers.append(CNNLSTMBlock(256, 256))
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=4, dropout=0.0, batch_first=True)

        #self.final_fc = nn.Sequential(nn.Linear(256, 14))
        self.final_fc = TimeDistributed(nn.Linear(256, 14), batch_first=True)
        self.dropout = Multisample_Dropout()

        self.init_weights()
        self.final_fc.apply(initialize_weights)

    def init_weights(self):
        for name, param in self.lstm_layers.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

    def forward(self, x, targets=None):
        
        for layer in self.lstm_layers:
            x = layer(x)
        x, _ = self.attention(x, x, x)
        x = self.dropout(x, self.final_fc)
        #x = self.final_fc(x)
        
        single_part = x[:, :, :output_single_num]
        single_part = torch.mean(single_part, axis=1)
        multi_part = x[:, :, output_single_num:]
        multi_part = multi_part.permute(0, 2, 1).reshape(-1, 360)

        logits = torch.concat([single_part, multi_part], axis=1)
        
        return logits


class LeapDataset(Dataset):
    def __init__(self, inputs_array, outputs_array=None, mode = "train"):
        super().__init__()
        
        self.mode = mode
        self.inputs = torch.from_numpy(inputs_array)
        if self.mode in ("train", "valid"):
            self.outputs = torch.from_numpy(outputs_array)
    
    def __getitem__(self, idx):

        aug_flip = False
        aug_flag = random.random()
        if aug_flag > 1:
            aug_flip = True
        inputs = self.inputs[idx].unsqueeze(0)

        single_input = torch.tile(inputs[:, :input_single_num].reshape(-1, 1, input_single_num),(1, 60, 1))
        series_input = inputs[:, input_single_num:].reshape(-1, input_series_num, 60).permute(0, 2, 1)
        if aug_flip and self.mode == "train":
            series_input = torch.flip(series_input, [1])
        inputs = torch.concat([single_input, series_input], axis=-1).squeeze(0)
        inputs = inputs.to(torch.float32)

        if self.mode in ("train", "valid"):
            outputs = self.outputs[idx]
            if aug_flip and self.mode == "train":
                outputs = torch.concat([
                    outputs[:output_single_num], 
                    torch.flip(outputs[ptend_index], [0]),
                    torch.flip(outputs[ptend_q0001_index], [0]),
                    torch.flip(outputs[ptend_q0002_index], [0]),
                    torch.flip(outputs[ptend_q0003_index], [0]),
                    torch.flip(outputs[ptend_u_index], [0]),
                    torch.flip(outputs[ptend_v_index], [0]),
                ], axis=-1)
            outputs = outputs.to(torch.float32)
            return inputs, outputs
        
        return inputs
    
    def __len__(self):
        return len(self.inputs)


    
def train_loop(train_loader, val_loader, model, optimizer,
               scheduler, criterion, score_func, gid):
    # logger.info(f'============fold:{fold} training==============')

    train_steps_per_epoch = len(train_loader)  # int(len(train_folds) / CFG.batch_size)

    num_train_steps = train_steps_per_epoch * CFG.epoch
    #    swa_steps = eval_steps

    # ====================================
    best_score_epoch = np.inf  # 打分得越低越好
    best_pred_epoch = -1
    best_epoch = -1
    es_flag = False
    early_step = 0
    best_zero_list = []

    optimizer.zero_grad()
    for epoch in range(CFG.epoch):

        eval_steps = get_evaluation_steps(train_steps_per_epoch, 8)
        eval_steps = eval_steps[-3:]
        
        model.train()
        avg_train = AverageMeter()
        start = time.time()

        for step, (inputs, labels) in tqdm(enumerate(train_loader), desc="training", total=len(train_loader), leave=True):
            inputs= inputs.to(CFG.device, non_blocking=True)
            labels = labels.to(CFG.device, non_blocking=True)

            train_batch_size = labels.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, labels, gid)

            avg_train.update(loss.item(), train_batch_size)

            loss.backward()
            if CFG.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

            scheduler.step(epoch + step/train_steps_per_epoch)
            
            # loss.val 当前batch的loss, loss.avg是epoch里面的avg
            if (CFG.print_freq > 0) and (step % CFG.print_freq == 0 or step == (train_steps_per_epoch - 1)) and step > 0:
                if CFG.scheduler_type == "plateau":
                    print('Epoch: [{0}][{1}/{2}] '
                          'Elapsed {remain:s} '
                          'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                          'Grad: {grad_norm:.4f}  '
                          .format(epoch + 1, step, len(train_loader),
                                  remain=timeSince(start, float(step + 1) / len(train_loader)),
                                  loss=avg_train,
                                  grad_norm=grad_norm))
                else:
                    print('Epoch: [{0}][{1}/{2}] '
                          'Elapsed {remain:s} '
                          'LR0: {lr0:.6f}  '
                          'LR1: {lr1:.6f}  '
                          'LR2: {lr2:.6f}  '
                          .format(epoch + 1, step, len(train_loader),
                                  remain=timeSince(start, float(step + 1) / len(train_loader)),
                                  lr0=optimizer.param_groups[0]['lr'],
                                  lr1=optimizer.param_groups[1]['lr'],
                                  lr2=optimizer.param_groups[2]['lr'],
                                 )) 

                    # lr3=optimizer.param_groups[3]['lr'],'LR3: {lr3:.6f}  '
                    # .get_last_lr()[0]
                    # 'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    # 'Grad: {grad_norm:.4f}  '
                    # loss=avg_train,
                    # grad_norm=grad_norm,

            # 多次打日志，一个epoch里train打印log
            if (step == (len(train_loader) - 1)) or (step + 1 in eval_steps):
                time_consume = time.time() - start

                if CFG.scheduler_type == "plateau":
                    logger.info(f"Epoch: [{epoch + 1}][{step + 1}/{len(train_loader)}], "
                                f"train_avg_loss:{avg_train.avg:.4f}, "
                                f"grad norm2:{grad_norm:.2f}, "
                                f"time:{time_consume:.2f}, ")
                else:
                    logger.info(f"Epoch: [{epoch + 1}][{step + 1}/{len(train_loader)}], "
                                f"train_avg_loss:{avg_train.avg:.4f}, "
                                f"grad norm2:{grad_norm:.2f}, "
                                f"time:{time_consume:.2f}, "
                    )
                    # #f"Lr:{scheduler.get_last_lr()[0]:.8f}"

            if CFG.eval_val and ((step + 1) in eval_steps):
                # eval环节用不到
                # 不能放到if判断条件外,每个batch都要删一次，gpu利用率拉不满
                del inputs, labels
                gc.collect()
                torch.cuda.empty_cache()

                # if CFG.use_swa and epoch >= CFG.swa_start_epoch:
                # optimizer.swap_swa_sgd()
                avg_val_loss, oof_score, oof_pred, zero_list_tmp, _ = valid_one_epoch(
                    model, val_loader, epoch, criterion, score_func, gid
                )
                #if epoch + 1 > 8:
                #    scheduler2.step(oof_score)
                
                model.train()

                if CFG.save_best_model:
                    if oof_score < best_score_epoch:
                        best_score_epoch = oof_score
                        best_pred_epoch = oof_pred
                        best_epoch = epoch
                        early_step = 0
                        if CFG.save_path is not None:
                            best_zero_list = zero_list_tmp
                            os.makedirs(CFG.save_path, exist_ok=True)
                            torch.save(
                                model.module.state_dict(),
                                os.path.join(CFG.save_path, f'{CFG.exp_id}_g{gid}.pt')
                            )
                            logger.info(f"############ saving model..., {best_zero_list} [[[score: {round(oof_score, 5)}]]] ############")
                            print(f"############ saving model..., {best_zero_list} [[[score: {round(oof_score, 5)}]]] ############")
                    else:
                        print(f"############ Not improved >_< ...... ############")
                        if CFG.early_stop_flag and early_step >= CFG.early_stop_step:
                            logger.info(f"early stop---loss:{best_score_epoch:.4f}, epoch:{best_epoch}")
                            print(f"early stop---loss:{best_score_epoch:.4f}, epoch:{best_epoch}")
                            es_flag = True
                            break
                        early_step += 1
                        
                            
        # if CFG.use_swa and epoch >= CFG.swa_start_epoch:
        # optimizer.swap_swa_sgd()

        gc.collect()
        torch.cuda.empty_cache()
        # early stop
        if es_flag:
            break

    if not es_flag:
        logger.info(f"Not stopped---loss:{best_score_epoch:.4f}, epoch:{best_epoch}")

    gc.collect()
    torch.cuda.empty_cache()
    return best_pred_epoch, best_score_epoch,best_zero_list

    
with open("../data/mean.json",'r') as f:
    mean_dict = json.load(f)

with open("../data/std.json",'r') as f:
    std_dict = json.load(f)

target_col_series_name = ["ptend_t","ptend_q0001","ptend_q0002","ptend_q0003","ptend_u","ptend_v"]
target_col_single = [
    'cam_out_NETSW',"cam_out_FLWDS","cam_out_PRECSC","cam_out_PRECC",
    "cam_out_SOLS","cam_out_SOLL","cam_out_SOLSD","cam_out_SOLLD"
]

target_col_series = []
for _ in target_col_series_name:
    target_col_series += [ _ + f"_{i}" for i in range(60)]
    
target_cols = target_col_single + target_col_series

non_pred_list = [
    'ptend_q0001_0', 'ptend_q0001_1', 'ptend_q0001_2', 'ptend_q0001_3',
    'ptend_q0001_4', 'ptend_q0001_5', 'ptend_q0001_6', 'ptend_q0001_7',
    'ptend_q0001_8', 'ptend_q0001_9', 'ptend_q0001_10', 'ptend_q0001_11',
    'ptend_q0002_0', 'ptend_q0002_1', 'ptend_q0002_2', 'ptend_q0002_3',
    'ptend_q0002_4', 'ptend_q0002_5', 'ptend_q0002_6', 'ptend_q0002_7',
    'ptend_q0002_8', 'ptend_q0002_9', 'ptend_q0002_10', 'ptend_q0002_11', 'ptend_q0003_0',
    'ptend_q0003_1', 'ptend_q0003_2', 'ptend_q0003_3', 'ptend_q0003_4',
    'ptend_q0003_5', 'ptend_q0003_6', 'ptend_q0003_7', 'ptend_q0003_8',
    'ptend_q0003_9', 'ptend_q0003_10', 'ptend_q0003_11', 'ptend_u_0',
    'ptend_u_1', 'ptend_u_2', 'ptend_u_3', 'ptend_u_4', 'ptend_u_5',
    'ptend_u_6', 'ptend_u_7', 'ptend_u_8', 'ptend_u_9', 'ptend_u_10',
    'ptend_u_11', 'ptend_v_0', 'ptend_v_1', 'ptend_v_2', 'ptend_v_3',
    'ptend_v_4', 'ptend_v_5', 'ptend_v_6', 'ptend_v_7', 'ptend_v_8',
    'ptend_v_9', 'ptend_v_10', 'ptend_v_11'
] 

sub_order = ['ptend_t_0','ptend_t_1','ptend_t_2','ptend_t_3','ptend_t_4','ptend_t_5','ptend_t_6','ptend_t_7','ptend_t_8','ptend_t_9','ptend_t_10','ptend_t_11','ptend_t_12','ptend_t_13','ptend_t_14','ptend_t_15','ptend_t_16','ptend_t_17','ptend_t_18','ptend_t_19','ptend_t_20','ptend_t_21','ptend_t_22','ptend_t_23','ptend_t_24','ptend_t_25','ptend_t_26','ptend_t_27','ptend_t_28','ptend_t_29','ptend_t_30','ptend_t_31','ptend_t_32','ptend_t_33','ptend_t_34','ptend_t_35','ptend_t_36','ptend_t_37','ptend_t_38','ptend_t_39','ptend_t_40','ptend_t_41','ptend_t_42','ptend_t_43','ptend_t_44','ptend_t_45','ptend_t_46','ptend_t_47','ptend_t_48','ptend_t_49','ptend_t_50','ptend_t_51','ptend_t_52','ptend_t_53','ptend_t_54','ptend_t_55','ptend_t_56','ptend_t_57','ptend_t_58','ptend_t_59','ptend_q0001_0','ptend_q0001_1','ptend_q0001_2','ptend_q0001_3','ptend_q0001_4','ptend_q0001_5','ptend_q0001_6','ptend_q0001_7','ptend_q0001_8','ptend_q0001_9','ptend_q0001_10','ptend_q0001_11','ptend_q0001_12','ptend_q0001_13','ptend_q0001_14','ptend_q0001_15','ptend_q0001_16','ptend_q0001_17','ptend_q0001_18','ptend_q0001_19','ptend_q0001_20','ptend_q0001_21','ptend_q0001_22','ptend_q0001_23','ptend_q0001_24','ptend_q0001_25','ptend_q0001_26','ptend_q0001_27','ptend_q0001_28','ptend_q0001_29','ptend_q0001_30','ptend_q0001_31','ptend_q0001_32','ptend_q0001_33','ptend_q0001_34','ptend_q0001_35',
 'ptend_q0001_36','ptend_q0001_37','ptend_q0001_38','ptend_q0001_39','ptend_q0001_40','ptend_q0001_41','ptend_q0001_42','ptend_q0001_43','ptend_q0001_44','ptend_q0001_45','ptend_q0001_46','ptend_q0001_47','ptend_q0001_48','ptend_q0001_49','ptend_q0001_50','ptend_q0001_51','ptend_q0001_52','ptend_q0001_53','ptend_q0001_54','ptend_q0001_55','ptend_q0001_56','ptend_q0001_57','ptend_q0001_58','ptend_q0001_59','ptend_q0002_0','ptend_q0002_1','ptend_q0002_2','ptend_q0002_3','ptend_q0002_4','ptend_q0002_5','ptend_q0002_6','ptend_q0002_7','ptend_q0002_8','ptend_q0002_9','ptend_q0002_10','ptend_q0002_11','ptend_q0002_12','ptend_q0002_13','ptend_q0002_14','ptend_q0002_15','ptend_q0002_16','ptend_q0002_17','ptend_q0002_18','ptend_q0002_19','ptend_q0002_20','ptend_q0002_21','ptend_q0002_22','ptend_q0002_23','ptend_q0002_24','ptend_q0002_25','ptend_q0002_26','ptend_q0002_27','ptend_q0002_28','ptend_q0002_29','ptend_q0002_30','ptend_q0002_31','ptend_q0002_32','ptend_q0002_33','ptend_q0002_34','ptend_q0002_35','ptend_q0002_36','ptend_q0002_37','ptend_q0002_38','ptend_q0002_39','ptend_q0002_40','ptend_q0002_41','ptend_q0002_42','ptend_q0002_43','ptend_q0002_44','ptend_q0002_45','ptend_q0002_46','ptend_q0002_47','ptend_q0002_48','ptend_q0002_49','ptend_q0002_50','ptend_q0002_51','ptend_q0002_52','ptend_q0002_53','ptend_q0002_54','ptend_q0002_55','ptend_q0002_56','ptend_q0002_57','ptend_q0002_58','ptend_q0002_59',
 'ptend_q0003_0','ptend_q0003_1','ptend_q0003_2','ptend_q0003_3','ptend_q0003_4','ptend_q0003_5','ptend_q0003_6','ptend_q0003_7','ptend_q0003_8','ptend_q0003_9','ptend_q0003_10','ptend_q0003_11','ptend_q0003_12','ptend_q0003_13','ptend_q0003_14','ptend_q0003_15','ptend_q0003_16','ptend_q0003_17','ptend_q0003_18','ptend_q0003_19','ptend_q0003_20','ptend_q0003_21','ptend_q0003_22','ptend_q0003_23','ptend_q0003_24','ptend_q0003_25','ptend_q0003_26','ptend_q0003_27','ptend_q0003_28','ptend_q0003_29','ptend_q0003_30','ptend_q0003_31','ptend_q0003_32','ptend_q0003_33','ptend_q0003_34','ptend_q0003_35','ptend_q0003_36','ptend_q0003_37','ptend_q0003_38','ptend_q0003_39','ptend_q0003_40','ptend_q0003_41','ptend_q0003_42','ptend_q0003_43','ptend_q0003_44','ptend_q0003_45','ptend_q0003_46','ptend_q0003_47','ptend_q0003_48','ptend_q0003_49','ptend_q0003_50','ptend_q0003_51','ptend_q0003_52','ptend_q0003_53','ptend_q0003_54','ptend_q0003_55','ptend_q0003_56','ptend_q0003_57','ptend_q0003_58','ptend_q0003_59','ptend_u_0','ptend_u_1','ptend_u_2','ptend_u_3','ptend_u_4','ptend_u_5','ptend_u_6','ptend_u_7','ptend_u_8','ptend_u_9','ptend_u_10','ptend_u_11','ptend_u_12','ptend_u_13','ptend_u_14','ptend_u_15','ptend_u_16','ptend_u_17','ptend_u_18','ptend_u_19','ptend_u_20','ptend_u_21','ptend_u_22','ptend_u_23','ptend_u_24','ptend_u_25','ptend_u_26','ptend_u_27','ptend_u_28','ptend_u_29','ptend_u_30','ptend_u_31','ptend_u_32','ptend_u_33','ptend_u_34','ptend_u_35','ptend_u_36','ptend_u_37','ptend_u_38','ptend_u_39','ptend_u_40','ptend_u_41','ptend_u_42','ptend_u_43','ptend_u_44','ptend_u_45','ptend_u_46','ptend_u_47','ptend_u_48','ptend_u_49','ptend_u_50','ptend_u_51','ptend_u_52','ptend_u_53','ptend_u_54','ptend_u_55','ptend_u_56','ptend_u_57','ptend_u_58','ptend_u_59',
 'ptend_v_0','ptend_v_1','ptend_v_2','ptend_v_3','ptend_v_4','ptend_v_5','ptend_v_6','ptend_v_7','ptend_v_8','ptend_v_9','ptend_v_10','ptend_v_11','ptend_v_12','ptend_v_13','ptend_v_14','ptend_v_15','ptend_v_16','ptend_v_17','ptend_v_18','ptend_v_19','ptend_v_20','ptend_v_21','ptend_v_22','ptend_v_23','ptend_v_24','ptend_v_25','ptend_v_26','ptend_v_27','ptend_v_28','ptend_v_29','ptend_v_30','ptend_v_31','ptend_v_32','ptend_v_33','ptend_v_34','ptend_v_35','ptend_v_36','ptend_v_37','ptend_v_38','ptend_v_39','ptend_v_40','ptend_v_41','ptend_v_42','ptend_v_43','ptend_v_44','ptend_v_45','ptend_v_46','ptend_v_47','ptend_v_48','ptend_v_49','ptend_v_50','ptend_v_51','ptend_v_52','ptend_v_53','ptend_v_54','ptend_v_55','ptend_v_56','ptend_v_57','ptend_v_58','ptend_v_59','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']

linear_corr_list = [f'ptend_q0002_{i}' for i in range(12,28)]

calc_loss_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 367]

GIDX_LOSS_DICT = {}
GIDX_LOSS_DICT[0] = np.arange(8)
for i in range(6):
    index = 8+i*60+np.arange(60)
    use_index = [ids for ids in index if ids in calc_loss_index]
    GIDX_LOSS_DICT[i+1] = np.array(use_index)
    print(f"gid {i+1} use {len(use_index)} to calc loss.")

zero_index = []
for _ in non_pred_list:
    zero_index.append(target_cols.index(_))

GIDX_DICT = {}
GIDX_DICT[0] = np.arange(8)
for i in range(6):
    GIDX_DICT[i+1] = 8+i*60+np.arange(60)



def negative_r2(y_true, y_pred, gid=None):
    score_list = []
    zero_list = []
    
    for i in range(y_true.shape[1]):
        score_ = r2_score(y_true[:, i], y_pred[:, i])
        # 线性相关
        if target_cols[i] in linear_corr_list:
            score_list.append(1.0)
            continue 
        # 其他列
        if score_ <= 0 and i not in zero_index:
            score_new = r2_score(y_true[:,i] * std_dict[target_cols[i]] + mean_dict[target_cols[i]], np.zeros(len(y_pred)))
            #logger.info(f"col name:{target_cols[i]}, score:{score_:.5f}, new score:{score_new:.5f}")
            if score_new > score_:
                score_ = score_new
                zero_list.append(target_cols[i])
        score_list.append(score_)

    base_scores = {}
    if gid is None:
        for gid in range(7):
            idx = GIDX_DICT[gid]
            #print(np.array(score_list)[idx])
            gid_score = np.mean(np.array(score_list)[idx])
            print(f'g{gid} r2: ', round(gid_score, 5))
            base_scores[gid] = gid_score
        return -np.mean(score_list), zero_list, base_scores
    else:
        idx = GIDX_DICT[gid]
        #print(np.array(score_list)[idx])
        return -np.mean(np.array(score_list)[idx]), [], base_scores



def valid_one_epoch(model, dataloader, epoch, criterion, score_func, gid=None):
    """
    model:模型
    dataloader: valid dataloader
    epoch: 当前epoch数
    criterion: torch版本的metrics preds, labels
    score_func: 计算concat完的打分,通常为sklearn版本的metrics, labels, pred{可能有时候loss和最终评分不是同一个}
    """
    model.eval()
    # torch.backends.cudnn.benchmark = True
    avg_eval = AverageMeter()
    oof_pred_list = []
    oof_label_list = []

    start = time.time()

    for step, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(CFG.device, non_blocking=True)
        labels = labels.to(CFG.device, non_blocking=True)

        batch_size_val = labels.size(0)

        with torch.no_grad():
            out_puts = model(inputs)
            loss = criterion(out_puts, labels, gid)

        avg_eval.update(loss.item(), batch_size_val)

        oof_pred_list.append(out_puts.detach().cpu().numpy())
        oof_label_list.append(labels.detach().cpu().numpy())

        # loss.val是当前的metric loss.avg是历史平均
        if (CFG.print_freq > 0) and (step % CFG.print_freq == 0 or step == (len(dataloader) - 1)) and step > 0:
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.6f}({loss.avg:.6f}) '
                  .format(step, len(dataloader),
                          loss=avg_eval,
                          remain=timeSince(start, float(step + 1) / len(dataloader))))

    oof_pred = np.concatenate(oof_pred_list, axis=0)
    oof_label = np.concatenate(oof_label_list, axis=0)
    # 如果有多个打分，可以在这里处理下，最后logger相应的做处理即可
    for _ in zero_index:
        col_ = target_cols[_]
        oof_pred[:, _] = -mean_dict[col_] / std_dict[col_]

    oof_score, zero_list_tmp, base_scores = score_func(oof_label, oof_pred, gid)
    time_consume = time.time() - start

    logger.info(f"epoch:{epoch+1},  "
                f"val_avg_loss: {avg_eval.avg:.6f},  "
                f"val_oof_score: {oof_score:.6f},  "
                f"time: {time_consume:.2f}")
    del oof_label, oof_pred_list, oof_label_list, inputs
    torch.cuda.empty_cache()
    gc.collect()

    return avg_eval.avg, oof_score, oof_pred, zero_list_tmp, base_scores


def pred_func(inputs_array_path, model_dir, model_new, new_zero_pred_list, bad_gids):
    part1_input = np.load(inputs_array_path)
    test_dataset = LeapDataset(part1_input, mode='test')
    
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=CFG.val_batch_size,
        num_workers=CFG.num_workers,
        shuffle=False, pin_memory=True, drop_last=False
    )
    
    oof_pred_llist=[]
    for gid in range(7):
        
        if gid in bad_gids:
            model_path = resume_path
        else:
            model_path = f"{model_dir}/{CFG.exp_id}_g{gid}.pt"
        print(f'load model from {model_path}')
        model = model_new
        model.load_state_dict(torch.load(model_path), strict=True)
        model = model.to(CFG.device)
        model.eval()
        oof_pred_list = []
        
        for step, inputs in tqdm(enumerate(dataloader), desc='test one epoch', total=len(dataloader)):
            inputs = inputs.to(CFG.device, non_blocking=True)
            with torch.no_grad():
                out_puts = model(inputs)
                out_puts = out_puts.to(torch.float64)
            oof_pred_list.append(out_puts.detach().cpu().numpy())

        oof_pred = np.concatenate(oof_pred_list, axis=0)[:, GIDX_DICT[gid]]
        oof_pred_llist.append(oof_pred)
    oof_pred = np.concatenate(oof_pred_llist, axis=1)

    sub_12_27 = pd.read_parquet("../data/sub_12_27.parquet")
    # 归一化回去
    final_np = np.zeros(oof_pred.shape, dtype=np.float64)
    sub_sample_old = pd.read_csv("../../raw_data/kaggle-data/sample_submission_old.csv", nrows=1)
    for idx, col in enumerate(sub_order):
        # 直接0
        if col in new_zero_pred_list:
            final_np[:,idx] = 0.0
            continue
        elif col in linear_corr_list:
            final_np[:,idx] = sub_12_27[col].values
        else: 
            old_idx = target_cols.index(col)
            final_np[:,idx] = (oof_pred[:,old_idx] * std_dict[col] + mean_dict[col]) / sub_sample_old[col].values[0]
            # print(col, sub_sample_old[col].values[0])
            
    sub_sample = pd.read_csv("../../raw_data/kaggle-data/sample_submission.csv")
    sub_sample.iloc[:,1:] = final_np
    sub_sample.to_parquet(f"../infer_outputs/Jo_ex912sep2_cv78935.parquet")
    sub_sample.to_parquet(f"../../submission/subs/Jo_ex912sep2_cv78935.parquet")


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
 
    part0_input = np.load("../data/train_inputs_final.npy")
    print(f"train inputs: {part0_input.shape}")
    part0_output = np.load("../data/train_outputs_final.npy")
    print(f"train outputs: {part0_output.shape}")

    part1_input = np.load("../data/valid_inputs.npy")
    print(f"valid inputs: {part1_input.shape}")
    part1_output = np.load("../data/valid_outputs.npy")
    print(f"valid outputs: {part1_output.shape}")
    
    gc.collect()
    logger.info(f"train inputs: {part0_input.shape}")
    logger.info(f"train outputs: {part0_output.shape}")
    logger.info(f"valid inputs: {part1_input.shape}")
    logger.info(f"valid outputs: {part1_output.shape}")

    valid_dataset = LeapDataset(part1_input, part1_output, mode='valid')
    del part1_input, part1_output
    gc.collect()
    logger.info(f"valid done")
    train_dataset = LeapDataset(part0_input, part0_output, mode='train')
    del part0_input, part0_output
    gc.collect()
    logger.info(f"train done")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        num_workers=CFG.num_workers,
        shuffle=True, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=CFG.val_batch_size,
        num_workers=CFG.num_workers,
        shuffle=False, pin_memory=True, drop_last=False
    )
    train_steps_per_epoch = len(train_loader)
    num_train_steps = train_steps_per_epoch * CFG.epoch

    resume_path = f'../outputs/ex912/ex912.pt'
    print(f"load model from {resume_path}")
    model = LEAP_Model().to(CFG.device)
    model.load_state_dict(torch.load(resume_path), strict=True)
    
    total_params = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    total_size_mb = total_size / 1024 / 1024
    print(f"Total number of parameters: {total_params}")
    print(f"Total size of parameters: {int(total_size_mb)} MB")
    logger.info(f"Total number of parameters: {total_params}")
    logger.info(f"Total size of parameters: {int(total_size_mb)} MB")

    #criterion = torch.nn.SmoothL1Loss()
    criterion = LEAPLoss(w=1)
    score_func = negative_r2
    avg_val_loss, oof_score, oof_pred, zero_list_tmp, base_scores = valid_one_epoch(
        model, val_loader, -1, criterion, score_func
    )

    bad_gids = []
    for gid in range(7):
        base_score = base_scores[gid]
        print('begin to finetune group target: ', gid, 'init score: *****', round(base_score, 5), '*****')
        print("")
        logger.info(f'begin to finetune group target: {gid}, init score: ***** {round(base_score, 5)} *****')

        model = LEAP_Model().to(CFG.device)
        model.load_state_dict(torch.load(resume_path), strict=True)
        model = nn.DataParallel(model.to(CFG.device), device_ids=list(range(3)))
        
        optimizer = AdamW([
            {'params': model.module.lstm_layers.parameters(), 'lr': CFG.lr_gru*0.2},
            {'params': model.module.attention.parameters(), 'lr': CFG.lr_attention*0.2},
            {'params': model.module.final_fc.parameters(), 'lr': CFG.lr_fc*0.2},
        ], weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1, T_mult=1, eta_min=2e-7)
        oof_npy, best_score, best_zero_list = train_loop(
            train_loader, val_loader, model, optimizer, scheduler, criterion, score_func, gid
        )
        
        if abs(best_score) < abs(base_score):
            bad_gids.append(gid)
        
        print(f"g{gid} R2 ############### {best_score} --- {base_score} ###############")
        logger.info(f"best_zero_list: {best_zero_list}")
        logger.info("###########################")
        logger.info(f"g{gid} R2 ####### {best_score} --- {base_score} #######")
        logger.info("###########################")
        non_pred_list += best_zero_list
        for _ in best_zero_list:
            idx = target_cols.index(_)
            oof_npy[:,idx] = -mean_dict[_]/std_dict[_]

        np.save(f"../outputs/{exp_id}/oof_g{gid}.npy", oof_npy[:,GIDX_DICT[gid]])
        #logger.info(f"r2 score: {best_score:.6f}")
        
        del model
        gc.collect()
        torch.cuda.empty_cache()

        print("")
        print("")
        print("")
    
    print("merge oof of each group")
    logger.info("merge oof of each group")
    oof_file_names = glob(f"../outputs/{exp_id}/oof_g*.npy")
    oof_file_names.sort()
    print(oof_file_names)
    oof_npy = np.concatenate([np.load(oof_file_name) for oof_file_name in oof_file_names], axis=-1)

    final_score, _, sep_scores = negative_r2(valid_dataset.outputs.numpy(), oof_npy)
    print(final_score, sep_scores)
    logger.info(f'{final_score}, {sep_scores}')

    tmp_model = LEAP_Model()
    print('oof_npy.shape', oof_npy.shape)
    logger.info(f'oof_npy.shape: {oof_npy.shape}')
    np.save(f"../outputs/{exp_id}/oof.npy", oof_npy)

    pred_func("../data/test_0_inputs.npy", CFG.save_path, tmp_model, non_pred_list, bad_gids)
        
    #logger.info(f"Mean folds score:{np.mean(overall_scores):.5f}")
    #logger.info(f"oof folds score:{score_func(train['score'],overall_oof):.5f}")


