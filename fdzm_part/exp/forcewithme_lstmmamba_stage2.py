from mamba_ssm import Mamba, Mamba2
# import torch
# batch, length, dim = 2, 64, 512
# x = torch.randn(batch, length, dim).to("cuda")
# model = Mamba(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=128,  # SSM state expansion factor
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
# y = model(x)

# batch, length, dim = 2, 64, 512
# x = torch.randn(batch, length, dim).to("cuda")
# model = Mamba2(
#     # This module uses roughly 3 * expand * d_model^2 parameters
#     d_model=dim, # Model dimension d_model
#     d_state=128,  # SSM state expansion factor, typically 64 or 128
#     d_conv=4,    # Local convolution width
#     expand=2,    # Block expansion factor
# ).to("cuda")
# y = model(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
import json
import gc
from sklearn.metrics import r2_score
from box import Box # pip install python-box
from torch.optim import AdamW,Adam
import time
from tqdm.auto import tqdm
import math
import sys
sys.path.insert(0, "../")
import os
import gc
from sklearn.metrics import r2_score
import functools
import polars as pl

from utils.model_utils import *
from utils.base_utils import *

# 只能在py文件里运行, 不能在Notebook运行
current_file_path = __file__
file_name = os.path.basename(current_file_path)
exp_id = file_name.split(".")[0]
# exp_id = "001"
os.makedirs(f"../outputs/{exp_id}",exist_ok=True)

config = {

    # ======================= global setting =====================
    "exp_id": exp_id,
    "loss_v": 1, # 0:MSE, 1:SmoothL1, 2: SmoothL1+R2
    "seed": 1999517,
    "log_path": f"../outputs/{exp_id}/log.txt",
    "save_path": f"../outputs/{exp_id}",
    "device": "cuda:0",# cuda:0
    "print_freq": 1000,
    "save_best_model": True,
    # ======================== train & val ==========================
    "epoch": 8,
    "gradient_checkpointing_enable": False,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 9,
    "evaluate_n_times_per_epoch": 2,
    "evaluate_n_times_per_epoch_last": 4,
    "last_epoch_num": 1,
    "early_stop_flag": False,
    "early_stop_step": 1, # 1的话其实已经是两个epoch不增长了
    "batch_size": 2048,
    "val_batch_size": 2048*2,
    "eval_val": True,
    "num_workers": 40,
    "apex": False,
    "debug": False,
    "scheduler_type": "cos_restart",
    "n_warmup_steps": 1,
    "cosine_n_cycles": 0.5,
    "learning_rate": 4e-4,
    "T_0": 4,
    "T_mult":1,
    "eta_min":1e-5,
    "resume": "../outputs/forcewithme_lstmmamba_stage1/forcewithme_lstmmamba_stage1.pt"
}
OFFSET = 8
columns_to_keep = ['sample_id']
CFG = Box(config)
logger = get_logger(filename=CFG.log_path, filemode='w', add_console=False)
seed_everything(seed=CFG.seed)

input_single_num = 16
input_series_num = 9
output_single_num = 8
output_series_num = 6

logger.info(f"exp id:{exp_id}")
logger.info(f"config:{config}")

class LeapDataset(Dataset):
    def __init__(self, inputs_array, outputs_array=None, mode = "train"):
        
        super().__init__()
        
        self.mode = mode
        #series_part = inputs_array[:,input_single_num:].reshape(-1,60,input_series_num, order="F")# b, 60, 9
        #single_part = np.tile(inputs_array[:,:input_single_num].reshape(-1,1,input_single_num),(1,60,1))# b,60,16
        
        # self.inputs = torch.from_numpy(
        #     np.concatenate([
        #         np.tile(inputs_array[:,:input_single_num].reshape(-1,1,input_single_num),(1,60,1)), # b,60,16
        #         inputs_array[:,input_single_num:].reshape(-1,60,input_series_num, order="F") # b, 60, 9
        #     ], axis = -1)
        #     )

        self.inputs = torch.from_numpy(inputs_array)
        
        if self.mode == "train":
            self.outputs = torch.from_numpy(outputs_array)
        
        
    def __getitem__(self, idx):
        inputs = self.inputs[idx].unsqueeze(0)

        inputs = torch.concat([
                torch.tile(inputs[:,:input_single_num].reshape(-1,1,input_single_num),(1,60,1)), # b,60,16
                inputs[:,input_single_num:].reshape(-1,input_series_num,60).permute(0,2,1) # b, 60, 9
            ], axis = -1).squeeze(0)
        
        inputs = inputs.to(torch.float32)

        if self.mode == "train":
            outputs = self.outputs[idx]
            outputs = outputs.to(torch.float32)

            return inputs, outputs
        return inputs
    
    def __len__(self):
        return len(self.inputs)

class LSTM_BLOCK(nn.Module):  
    def __init__(self, input_dim=512, embed_dim=256, num_lstm=1, dr_mlp=0., dr_lstm=0., forward_expansion=2, first_layer=False):  
        super(LSTM_BLOCK, self).__init__()  
        self.embed_dim = embed_dim  
        self.first_layer = first_layer
        
        # 多头自注意力模块  
        self.lstm_layer = nn.LSTM(input_size=input_dim, 
                                            hidden_size=embed_dim, num_layers=num_lstm, bidirectional=True, batch_first=True)
  
        # 前馈神经网络  
        self.feed_forward = nn.Sequential(  
            nn.Linear(embed_dim*2, forward_expansion * embed_dim*2),  
            nn.GELU(),  
            nn.Linear(forward_expansion * embed_dim*2, embed_dim*2)  
        )  
  
        # 层归一化  
        self.norm1 = nn.LayerNorm([60, embed_dim*2])  
        self.norm2 = nn.LayerNorm([60, embed_dim*2])  
  
        # Dropout  
        self.dropout_mlp = nn.Dropout(dr_mlp)  
        self.dropout_lstm = nn.Dropout(dr_lstm)  
  
    def forward(self, x):  
        # 残差连接和层归一化  
        outputs, _ = self.lstm_layer(x)  
        if not self.first_layer:
            outputs = self.norm1(x + self.dropout_lstm(outputs))  
        else:
            outputs = self.norm1(outputs)

        # 前馈神经网络  
        ff_output = self.feed_forward(outputs)  
        outputs = self.norm2(outputs + self.dropout_mlp(ff_output))  
        return outputs  

class LeapModel(nn.Module):
    def __init__(self, inputs_dim=25, num_lstm=3, hidden_state=256, num_mamba=2, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.output_single_num = 8
        
        self.enc_norm = nn.LayerNorm(512)
        self.enc_act = nn.GELU()
        self.lstm_layers = nn.ModuleDict()
        self.lstm_enc = nn.LSTM(input_size=inputs_dim, hidden_size=hidden_state, num_layers=1, bidirectional=True, batch_first=True)
        # 定义LSTM层
        for i in range(num_lstm):  # 假设有5个LSTM层
            lstm_key = f'lstm{i + 1}'
            self.lstm_layers[lstm_key] = LSTM_BLOCK(hidden_state*2, num_lstm=2,
                                               embed_dim=hidden_state, first_layer=False)
        self.mamba_layers = nn.ModuleDict()
        for i in range(num_mamba): 
            self.mamba_layers[f'mamba{i+1}'] = Mamba(
                                        # This module uses roughly 3 * expand * d_model^2 parameters
                                        d_model=hidden_state*2, # Model dimension d_model
                                        d_state=d_state,  # SSM state expansion factor
                                        d_conv=d_conv,    # Local convolution width
                                        expand=expand,    # Block expansion factor
                                    )

        self.fc = nn.Sequential(
            nn.Linear(hidden_state*2, 14),
            # nn.ReLU()  # 可选，FC之后的激活函数
        )

    def forward(self, inputs):
        outputs, _ = self.lstm_enc(inputs)
        
        # 逐层通过LSTM并应用残差连接
        for i, lstm_layer in enumerate(self.lstm_layers.values()):
            outputs = lstm_layer(outputs)

        mamba_outputs = outputs
        for i, mamba_layer in enumerate(self.mamba_layers.values()):
            mamba_outputs = mamba_layer(mamba_outputs)

        # outputs = torch.concat([mamba_outputs, outputs], dim=-1)
        outputs = self.fc(mamba_outputs) # b,60,14
        
        single_part = outputs[:,:,:self.output_single_num]
        single_part = torch.mean(single_part, axis=1) # b,8
        series_part = outputs[:,:,self.output_single_num:]
        series_part = series_part.permute(0,2,1).reshape(-1,360) # b,360

        outputs = torch.concat([single_part, series_part], axis=1)
        return outputs

def train_loop(train_loader, val_loader, model, optimizer,
               scheduler, criterion, score_func):
    # logger.info(f'============fold:{fold} training==============')

    train_steps_per_epoch = len(train_loader)  # int(len(train_folds) / CFG.batch_size)

    num_train_steps = train_steps_per_epoch * CFG.epoch

    # 评估下在哪几步需要评估效果
    eval_steps = get_evaluation_steps(train_steps_per_epoch,
                                      CFG.evaluate_n_times_per_epoch)
    #    swa_steps = eval_steps

    # ====================================
    best_score_epoch = np.inf  # 打分得越低越好
    best_pred_epoch = -1

    best_epoch = -1

    early_step = 0
    best_zero_list = []

    for epoch in range(CFG.epoch):
        if epoch == CFG.epoch - CFG.last_epoch_num:
            eval_steps = get_evaluation_steps(train_steps_per_epoch,
                                      CFG.evaluate_n_times_per_epoch_last)
        es_flag = False
        best_score_step = np.inf  # 打分得越低越好
        best_pred_step = -1

        model.train()
        avg_train = AverageMeter()

        start = time.time()

        optimizer.zero_grad()
        for step, (inputs, labels) in tqdm(enumerate(train_loader), desc='train_one_epoch', total=len(train_loader)):
            inputs= inputs.to(CFG.device, non_blocking=True)
            labels = labels.to(CFG.device, non_blocking=True)

            train_batch_size = labels.size(0)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            OFFSET = 8
            for i in range(6):
                output_diff =outputs[:,60*i+1+OFFSET:60*(i+1)+OFFSET] - outputs[:,60*i+OFFSET:60*(i+1)-1+OFFSET]  # 9:67 ~ 8:66   
                label_diff=labels[:,60*i+1+OFFSET:60*(i+1)+OFFSET]-labels[:,60*i+OFFSET:60*(i+1)-1+OFFSET] 
                loss += criterion(output_diff,label_diff)/6

            avg_train.update(loss.item(), train_batch_size)

            loss.backward()
            if CFG.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            optimizer.step()
            scheduler.step(epoch + step / train_steps_per_epoch)
            optimizer.zero_grad()

            # loss.val 当前batch的loss, loss.avg是epoch里面的avg
            # print出来的结果
            if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
                print('Epoch: [{0}][{1}/{2}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                      'Grad: {grad_norm:.4f}  '
                      'LR: {lr:.8f}  '
                      .format(epoch + 1, step, len(train_loader),
                              remain=timeSince(start, float(step + 1) / len(train_loader)),
                              loss=avg_train,
                              grad_norm=grad_norm,
                              lr=scheduler.get_lr()[0]))

            # 多次打日志，一个epoch里train打印log
            if (step == (len(train_loader) - 1)) or \
                    (step + 1 in eval_steps):
                time_consume = time.time() - start

                logger.info(f"Epoch: [{epoch}][{step + 1}/{len(train_loader)}], "
                            f"train_avg_loss:{avg_train.avg:.4f}, "
                            f"grad norm2:{grad_norm:.2f}, "
                            f"time:{time_consume:.2f}, "
                            f"Lr:{scheduler.get_last_lr()[0]:.8f}")

            
            if CFG.eval_val and ((step + 1) in eval_steps):
                # eval环节用不到
                # 不能放到if判断条件外,每个batch都要删一次，gpu利用率拉不满
                del inputs, labels
                gc.collect()
                torch.cuda.empty_cache()

                #                 if CFG.use_swa and epoch >= CFG.swa_start_epoch:
                #                     optimizer.swap_swa_sgd()
                avg_val_loss, oof_score, oof_pred, zero_list_tmp = valid_one_epoch(model, val_loader, epoch, criterion, score_func)
                model.train()

                if CFG.save_best_model:
                    if oof_score < best_score_step:
                        best_score_step = oof_score
                        best_pred_step = oof_pred
                        # best_epoch = epoch
                        if CFG.save_path is not None and oof_score < best_score_epoch:
                            best_zero_list = zero_list_tmp
                            os.makedirs(CFG.save_path, exist_ok=True)
                            torch.save(model.state_dict(),
                                       os.path.join(CFG.save_path, f'{CFG.exp_id}.pt'))
                            logger.info(f"saving model...,{best_zero_list}")
        #                 if CFG.use_swa and epoch >= CFG.swa_start_epoch:
        #                     optimizer.swap_swa_sgd()

    
        gc.collect()
        torch.cuda.empty_cache()
        # early stop
        if best_score_step < best_score_epoch:
            best_pred_epoch = best_pred_step
            best_score_epoch = best_score_step
            best_epoch = epoch
            early_step = 0
        else:
            if CFG.early_stop_flag:
                if early_step >= CFG.early_stop_step:
                    logger.info(f"early stop---loss:{best_score_epoch:.4f}, epoch:{best_epoch}")
                    es_flag = True
                    break
                else:
                    early_step += 1
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
target_col_single = ['cam_out_NETSW',"cam_out_FLWDS","cam_out_PRECSC","cam_out_PRECC","cam_out_SOLS",
                    "cam_out_SOLL","cam_out_SOLSD","cam_out_SOLLD"]

target_col_series = []
for _ in target_col_series_name:
    target_col_series += [ _ + f"_{i}" for i in range(60)]
    
target_cols = target_col_single + target_col_series


non_pred_list = ['ptend_q0001_0', 'ptend_q0001_1', 'ptend_q0001_2', 'ptend_q0001_3',
       'ptend_q0001_4', 'ptend_q0001_5', 'ptend_q0001_6', 'ptend_q0001_7',
       'ptend_q0001_8', 'ptend_q0001_9', 'ptend_q0001_10', 'ptend_q0001_11',
       'ptend_q0002_0', 'ptend_q0002_1', 'ptend_q0002_2', 'ptend_q0002_3',
       'ptend_q0002_4', 'ptend_q0002_5', 'ptend_q0002_6', 'ptend_q0002_7',
       'ptend_q0002_8', 'ptend_q0002_9', 'ptend_q0002_10', 'ptend_q0002_11',
       #'ptend_q0002_12', #'ptend_q0002_13', 'ptend_q0002_14', 
                 'ptend_q0003_0',
       'ptend_q0003_1', 'ptend_q0003_2', 'ptend_q0003_3', 'ptend_q0003_4',
       'ptend_q0003_5', 'ptend_q0003_6', 'ptend_q0003_7', 'ptend_q0003_8',
       'ptend_q0003_9', 'ptend_q0003_10', 'ptend_q0003_11', 'ptend_u_0',
       'ptend_u_1', 'ptend_u_2', 'ptend_u_3', 'ptend_u_4', 'ptend_u_5',
       'ptend_u_6', 'ptend_u_7', 'ptend_u_8', 'ptend_u_9', 'ptend_u_10',
       'ptend_u_11', 'ptend_v_0', 'ptend_v_1', 'ptend_v_2', 'ptend_v_3',
       'ptend_v_4', 'ptend_v_5', 'ptend_v_6', 'ptend_v_7', 'ptend_v_8',
       'ptend_v_9', 'ptend_v_10', 'ptend_v_11'] 

sub_order = ['ptend_t_0','ptend_t_1','ptend_t_2','ptend_t_3','ptend_t_4','ptend_t_5','ptend_t_6','ptend_t_7','ptend_t_8','ptend_t_9','ptend_t_10','ptend_t_11','ptend_t_12','ptend_t_13','ptend_t_14','ptend_t_15','ptend_t_16','ptend_t_17','ptend_t_18','ptend_t_19','ptend_t_20','ptend_t_21','ptend_t_22','ptend_t_23','ptend_t_24','ptend_t_25','ptend_t_26','ptend_t_27','ptend_t_28','ptend_t_29','ptend_t_30','ptend_t_31','ptend_t_32','ptend_t_33','ptend_t_34','ptend_t_35','ptend_t_36','ptend_t_37','ptend_t_38','ptend_t_39','ptend_t_40','ptend_t_41','ptend_t_42','ptend_t_43','ptend_t_44','ptend_t_45','ptend_t_46','ptend_t_47','ptend_t_48','ptend_t_49','ptend_t_50','ptend_t_51','ptend_t_52','ptend_t_53','ptend_t_54','ptend_t_55','ptend_t_56','ptend_t_57','ptend_t_58','ptend_t_59','ptend_q0001_0','ptend_q0001_1','ptend_q0001_2','ptend_q0001_3','ptend_q0001_4','ptend_q0001_5','ptend_q0001_6','ptend_q0001_7','ptend_q0001_8','ptend_q0001_9','ptend_q0001_10','ptend_q0001_11','ptend_q0001_12','ptend_q0001_13','ptend_q0001_14','ptend_q0001_15','ptend_q0001_16','ptend_q0001_17','ptend_q0001_18','ptend_q0001_19','ptend_q0001_20','ptend_q0001_21','ptend_q0001_22','ptend_q0001_23','ptend_q0001_24','ptend_q0001_25','ptend_q0001_26','ptend_q0001_27','ptend_q0001_28','ptend_q0001_29','ptend_q0001_30','ptend_q0001_31','ptend_q0001_32','ptend_q0001_33','ptend_q0001_34','ptend_q0001_35',
 'ptend_q0001_36','ptend_q0001_37','ptend_q0001_38','ptend_q0001_39','ptend_q0001_40','ptend_q0001_41','ptend_q0001_42','ptend_q0001_43','ptend_q0001_44','ptend_q0001_45','ptend_q0001_46','ptend_q0001_47','ptend_q0001_48','ptend_q0001_49','ptend_q0001_50','ptend_q0001_51','ptend_q0001_52','ptend_q0001_53','ptend_q0001_54','ptend_q0001_55','ptend_q0001_56','ptend_q0001_57','ptend_q0001_58','ptend_q0001_59','ptend_q0002_0','ptend_q0002_1','ptend_q0002_2','ptend_q0002_3','ptend_q0002_4','ptend_q0002_5','ptend_q0002_6','ptend_q0002_7','ptend_q0002_8','ptend_q0002_9','ptend_q0002_10','ptend_q0002_11','ptend_q0002_12','ptend_q0002_13','ptend_q0002_14','ptend_q0002_15','ptend_q0002_16','ptend_q0002_17','ptend_q0002_18','ptend_q0002_19','ptend_q0002_20','ptend_q0002_21','ptend_q0002_22','ptend_q0002_23','ptend_q0002_24','ptend_q0002_25','ptend_q0002_26','ptend_q0002_27','ptend_q0002_28','ptend_q0002_29','ptend_q0002_30','ptend_q0002_31','ptend_q0002_32','ptend_q0002_33','ptend_q0002_34','ptend_q0002_35','ptend_q0002_36','ptend_q0002_37','ptend_q0002_38','ptend_q0002_39','ptend_q0002_40','ptend_q0002_41','ptend_q0002_42','ptend_q0002_43','ptend_q0002_44','ptend_q0002_45','ptend_q0002_46','ptend_q0002_47','ptend_q0002_48','ptend_q0002_49','ptend_q0002_50','ptend_q0002_51','ptend_q0002_52','ptend_q0002_53','ptend_q0002_54','ptend_q0002_55','ptend_q0002_56','ptend_q0002_57','ptend_q0002_58','ptend_q0002_59',
 'ptend_q0003_0','ptend_q0003_1','ptend_q0003_2','ptend_q0003_3','ptend_q0003_4','ptend_q0003_5','ptend_q0003_6','ptend_q0003_7','ptend_q0003_8','ptend_q0003_9','ptend_q0003_10','ptend_q0003_11','ptend_q0003_12','ptend_q0003_13','ptend_q0003_14','ptend_q0003_15','ptend_q0003_16','ptend_q0003_17','ptend_q0003_18','ptend_q0003_19','ptend_q0003_20','ptend_q0003_21','ptend_q0003_22','ptend_q0003_23','ptend_q0003_24','ptend_q0003_25','ptend_q0003_26','ptend_q0003_27','ptend_q0003_28','ptend_q0003_29','ptend_q0003_30','ptend_q0003_31','ptend_q0003_32','ptend_q0003_33','ptend_q0003_34','ptend_q0003_35','ptend_q0003_36','ptend_q0003_37','ptend_q0003_38','ptend_q0003_39','ptend_q0003_40','ptend_q0003_41','ptend_q0003_42','ptend_q0003_43','ptend_q0003_44','ptend_q0003_45','ptend_q0003_46','ptend_q0003_47','ptend_q0003_48','ptend_q0003_49','ptend_q0003_50','ptend_q0003_51','ptend_q0003_52','ptend_q0003_53','ptend_q0003_54','ptend_q0003_55','ptend_q0003_56','ptend_q0003_57','ptend_q0003_58','ptend_q0003_59','ptend_u_0','ptend_u_1','ptend_u_2','ptend_u_3','ptend_u_4','ptend_u_5','ptend_u_6','ptend_u_7','ptend_u_8','ptend_u_9','ptend_u_10','ptend_u_11','ptend_u_12','ptend_u_13','ptend_u_14','ptend_u_15','ptend_u_16','ptend_u_17','ptend_u_18','ptend_u_19','ptend_u_20','ptend_u_21','ptend_u_22','ptend_u_23','ptend_u_24','ptend_u_25','ptend_u_26','ptend_u_27','ptend_u_28','ptend_u_29','ptend_u_30','ptend_u_31','ptend_u_32','ptend_u_33','ptend_u_34','ptend_u_35','ptend_u_36','ptend_u_37','ptend_u_38','ptend_u_39','ptend_u_40','ptend_u_41','ptend_u_42','ptend_u_43','ptend_u_44','ptend_u_45','ptend_u_46','ptend_u_47','ptend_u_48','ptend_u_49','ptend_u_50','ptend_u_51','ptend_u_52','ptend_u_53','ptend_u_54','ptend_u_55','ptend_u_56','ptend_u_57','ptend_u_58','ptend_u_59',
 'ptend_v_0','ptend_v_1','ptend_v_2','ptend_v_3','ptend_v_4','ptend_v_5','ptend_v_6','ptend_v_7','ptend_v_8','ptend_v_9','ptend_v_10','ptend_v_11','ptend_v_12','ptend_v_13','ptend_v_14','ptend_v_15','ptend_v_16','ptend_v_17','ptend_v_18','ptend_v_19','ptend_v_20','ptend_v_21','ptend_v_22','ptend_v_23','ptend_v_24','ptend_v_25','ptend_v_26','ptend_v_27','ptend_v_28','ptend_v_29','ptend_v_30','ptend_v_31','ptend_v_32','ptend_v_33','ptend_v_34','ptend_v_35','ptend_v_36','ptend_v_37','ptend_v_38','ptend_v_39','ptend_v_40','ptend_v_41','ptend_v_42','ptend_v_43','ptend_v_44','ptend_v_45','ptend_v_46','ptend_v_47','ptend_v_48','ptend_v_49','ptend_v_50','ptend_v_51','ptend_v_52','ptend_v_53','ptend_v_54','ptend_v_55','ptend_v_56','ptend_v_57','ptend_v_58','ptend_v_59','cam_out_NETSW','cam_out_FLWDS','cam_out_PRECSC','cam_out_PRECC','cam_out_SOLS','cam_out_SOLL','cam_out_SOLSD','cam_out_SOLLD']

linear_corr_list = [f'ptend_q0002_{i}' for i in range(12,28)]

zero_index = []
for _ in non_pred_list:
    zero_index.append(target_cols.index(_))

def negative_r2(y_true,y_pred):
    score_list = []
    zero_list = []
    
    for i in range(y_true.shape[1]):
        score_ = r2_score(y_true[:,i], y_pred[:,i])
        # 线性相关
        if target_cols[i] in linear_corr_list:
            score_list.append(1.0)
            continue 
        # 其他列
        if score_ <= 0 and i not in zero_index:
            score_new = r2_score(y_true[:,i] * std_dict[target_cols[i]] + mean_dict[target_cols[i]], np.zeros(len(y_pred)))
            logger.info(f"col name:{target_cols[i]}, score:{score_:.5f}, new score:{score_new:.5f}")
            if score_new > score_:
                score_ = score_new
                zero_list.append(target_cols[i])
        score_list.append(score_)

    return - np.mean(score_list), zero_list

def valid_one_epoch(model, dataloader, epoch, criterion, score_func):
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

    for step, (inputs, labels) in tqdm(enumerate(dataloader), desc='valid one epoch', total=len(dataloader)):
        inputs= inputs.to(CFG.device, non_blocking=True)
        labels = labels.to(CFG.device, non_blocking=True)

        batch_size_val = labels.size(0)

        with torch.no_grad():
            out_puts = model(inputs)
            loss = criterion(out_puts, labels)

        avg_eval.update(loss.item(), batch_size_val)

        oof_pred_list.append(out_puts.detach().cpu().numpy())
        oof_label_list.append(labels.detach().cpu().numpy())

        # loss.val是当前的metric loss.avg是历史平均
        if step % CFG.print_freq == 0 or step == (len(dataloader) - 1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(dataloader),
                          loss=avg_eval,
                          remain=timeSince(start, float(step + 1) / len(dataloader))))

    oof_pred = np.concatenate(oof_pred_list, axis=0)
    oof_label = np.concatenate(oof_label_list, axis=0)
    # 如果有多个打分，可以在这里处理下，最后logger相应的做处理即可
    for _ in zero_index:
        col_ = target_cols[_]
        oof_pred[:,_] = -mean_dict[col_]/std_dict[col_]

    oof_score, zero_list_tmp = score_func(oof_label, oof_pred)
    time_consume = time.time() - start

    logger.info(f"epoch:{epoch},  "
                f"val_avg_loss:{avg_eval.avg:.4f},  "
                f"val_oof_score:{oof_score:.4f},  "
                f"time:{time_consume:.2f}")
    del oof_label, oof_pred_list, oof_label_list, inputs
    torch.cuda.empty_cache()
    gc.collect()

    return avg_eval.avg, oof_score, oof_pred,zero_list_tmp


def pred_func(inputs_array_path,model_path,model_new,new_zero_pred_list):
    part1_input =np.load(inputs_array_path)
    test_dataset = LeapDataset(part1_input, mode='test')
    
    dataloader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=CFG.val_batch_size,
                                            num_workers=CFG.num_workers,
                                            shuffle=False, pin_memory=True, drop_last=False)
    model = model_new
    model = model.to(CFG.device)
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    model.load_state_dict(torch.load(model_path), strict=True)
    
    
    model.eval()
    oof_pred_list = []

    for step, inputs in tqdm(enumerate(dataloader), desc='test one epoch', total=len(dataloader)):
        inputs= inputs.to(CFG.device, non_blocking=True)

        with torch.no_grad():
            out_puts = model(inputs)
            out_puts = out_puts.to(torch.float64)
        oof_pred_list.append(out_puts.detach().cpu().numpy())

    oof_pred = np.concatenate(oof_pred_list, axis=0)
    
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
    columns_to_convert = [col for col in sub_sample.columns if col not in columns_to_keep]
    sub_sample[columns_to_convert] = sub_sample[columns_to_convert].astype(np.float64)
    sub_sample.iloc[:,1:] = final_np
    pl.from_pandas(sub_sample).write_parquet(f"../outputs/{exp_id}/exp{exp_id}_new.parquet")

def check_data_size(npy_files, ratio):
    # 预先读取所有文件的形状和总大小
    shapes = [np.load(file, mmap_mode='r').shape for file in npy_files]
    idxs = [get_shuffle_list(shape[0], ratio=ratio) for i,shape in enumerate(shapes)]
    total_size = sum(len(idx) for idx in idxs)
    print(total_size)




def get_shuffle_list(size, ratio):
    indices = np.arange(size)
    np.random.shuffle(indices)
    return indices[:int(size * ratio)]

def load_npy_files_efficiently(npy_files, ratio, given_idxs=None, max_workers=16):
    # 预先读取所有文件的形状和总大小
    shapes = [np.load(file, mmap_mode='r').shape for file in npy_files]
    if given_idxs:
        idxs = given_idxs
    else:
        idxs = [get_shuffle_list(shape[0], ratio=ratio) for i,shape in enumerate(shapes)]
    total_size = sum(len(idx) for idx in idxs)
    print(total_size)

    # 创建一个大的数组用于存储所有数据
    concatenated_array = np.empty((total_size,) + shapes[0][1:], dtype=np.float32)

    def load_and_assign(file, idx, start_position):
        data = np.load(file, mmap_mode='r')
        data = data[idx, :]
        # print(file, len(data))
        concatenated_array[start_position:start_position + len(idx)] = data
        return len(idx)

    # 使用多线程并行读取和处理数据
    current_position = 0
    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for file, idx in zip(npy_files, idxs):
            futures.append(executor.submit(load_and_assign, file, idx, current_position))
            current_position += len(idx)

        for future in tqdm(as_completed(futures), total=len(futures)):
            future.result()  # 等待所有任务完成

    if given_idxs:
        return concatenated_array
    else:
        return concatenated_array, idxs



use_year_month = [
             '01_02', '01_03', '01_04', '01_05', '01_06', '01_07', '01_08', '01_09', '01_10', '01_11', '01_12',
    '02_01', '02_02', '02_03', '02_04', '02_05', '02_06', '02_07', '02_08', '02_09', '02_10', '02_11', '02_12',
    '03_01', '03_02', '03_03', '03_04', '03_05', '03_06', '03_07', '03_08', '03_09', '03_10', '03_11', '03_12',
    '04_01', '04_02', '04_03', '04_04', '04_05', '04_06', '04_07', '04_08', '04_09', '04_10', '04_11', '04_12',
    '05_01', '05_02', '05_03', '05_04', '05_05', '05_06', '05_07', '05_08', '05_09', '05_10', '05_11', '05_12',
    '06_01', '06_02', '06_03', '06_04', '06_05', '06_06', '06_07', '06_08', '06_09', '06_10', '06_11', '06_12',
    '07_01', '07_02', '07_03', '07_04', '07_05', '07_06', '07_07', '07_08', '07_09', '07_10', '07_11', '07_12',
    '08_01', '08_02', '08_03', '08_04', '08_05', '08_06'
]
# use_year_month = use_year_month[:16]
base_dir = '../data/months'
train_inputs_files = [f"{base_dir}/train_{ym}_inputs.npy" for ym in use_year_month]
train_outputs_files = [f"{base_dir}/train_{ym}_outputs.npy" for ym in use_year_month]

if __name__ == "__main__":
    if CFG.debug:
        part0_input = np.load("../data/valid_inputs.npy")
        part0_output = np.load("../data/valid_outputs.npy")
        print(f"train outputs: {part0_output.shape}")

        part1_input = np.load("../data/valid_inputs.npy")
        part1_output = np.load("../data/valid_outputs.npy")
        print(f"valid outputs: {part1_output.shape}") 
    else:
        part0_input, idxs = load_npy_files_efficiently(train_inputs_files, ratio=1.0, max_workers=32)
        print(f"train outputs: {part0_input.shape}")
        part0_output = load_npy_files_efficiently(train_outputs_files, ratio=1.0, given_idxs=idxs, max_workers=32)
        print(f"train outputs: {part0_output.shape}")

        part1_input = np.load("../data/valid_inputs.npy")
        print(f"valid outputs: {part1_input.shape}")
        part1_output = np.load("../data/valid_outputs.npy")
        print(f"valid outputs: {part1_output.shape}")
    
    gc.collect()
    logger.info(f"train inputs:{part0_input.shape}")
    logger.info(f"train outputs:{part0_output.shape}")
    logger.info(f"valid inputs:{part1_input.shape}")
    logger.info(f"valid outputs:{part1_output.shape}")

    valid_dataset = LeapDataset(part1_input,part1_output)
    del part1_input,part1_output
    gc.collect()
    logger.info(f"valid done")
    train_dataset = LeapDataset(part0_input,part0_output)
    del part0_input,part0_output
    gc.collect()
    logger.info(f"train done")

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=CFG.batch_size,
                                            num_workers=CFG.num_workers,
                                            shuffle=True, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(valid_dataset,
                                            batch_size=CFG.val_batch_size,
                                            num_workers=CFG.num_workers,
                                            shuffle=False, pin_memory=True, drop_last=False)
    train_steps_per_epoch = len(train_loader)  # int(len(train_folds) / CFG.batch_size)

    num_train_steps = train_steps_per_epoch * CFG.epoch

    # 这里如果改模型需要改两次，因为要load存下来的权重
    model = LeapModel().to(CFG.device)
    model = nn.DataParallel(model, device_ids=[0,1,2,3])
    if CFG.resume is not None:
        weight = torch.load(CFG.resume)
        weight.pop("module.fc.0.weight")
        model.load_state_dict(weight, strict=False)
    tmp_model = LeapModel()

    optimizer = AdamW(model.parameters(), lr=CFG.learning_rate, betas=(0.9, 0.95))
    scheduler = get_scheduler(optimizer, CFG, num_train_steps=num_train_steps, loader=train_loader)
    if CFG.loss_v == 0:
        criterion = torch.nn.MSELoss()
    elif CFG.loss_v == 1:
        criterion = torch.nn.SmoothL1Loss()
    score_func = negative_r2

    oof_npy, best_score, best_zero_list = train_loop(train_loader, val_loader, model, optimizer,
               scheduler, criterion, score_func)
    logger.info(f"best_zero_list:{best_zero_list}")
    
    non_pred_list += best_zero_list
    for _ in best_zero_list:
        idx = target_cols.index(_)
        oof_npy[:,idx] = -mean_dict[_]/std_dict[_]

    np.save(f"../outputs/{exp_id}/oof.npy",oof_npy)
    logger.info(f"r2 score:{best_score:.6f}")

    pred_func(f"../data/test_0_inputs.npy",os.path.join(CFG.save_path, f'{CFG.exp_id}.pt'),tmp_model,non_pred_list)
    #logger.info(f"Mean folds score:{np.mean(overall_scores):.5f}")
    #logger.info(f"oof folds score:{score_func(train['score'],overall_oof):.5f}")