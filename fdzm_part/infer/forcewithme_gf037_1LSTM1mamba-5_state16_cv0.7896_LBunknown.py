from mamba_ssm import Mamba, Mamba2
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
import polars as pl
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
sys.path.insert(0, "../")
import os
import gc
from sklearn.metrics import r2_score
import functools
from glob import glob
from utils.model_utils import *
from utils.base_utils import *
np.set_printoptions(precision=5, linewidth=200)
# 只能在py文件里运行, 不能在Notebook运行
current_file_path = __file__
file_name = os.path.basename(current_file_path)
exp_id = file_name.split(".")[0]
# exp_id = "001"
os.makedirs(f"../infer_outputs/{exp_id}",exist_ok=True)
columns_to_keep = ['sample_id']
config = {
    # ======================= global setting =====================
    "exp_id": exp_id,
    "seed": 1999517,
    "log_path": f"../infer_outputs/{exp_id}/log.txt",
    "save_path": f"../weights/{exp_id}",
    "device": "cuda:0",# cuda:0
    "print_freq": 1000,
    "save_best_model": True,
    # ======================== train & val ==========================
    "epoch": 1,
    "gradient_checkpointing_enable": False,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 9,
    "evaluate_n_times_per_epoch": 1,
    "early_stop_flag": True,
    "early_stop_step": 1000, # 1的话其实已经是两个epoch不增长了
    "batch_size": 4032,
    "val_batch_size": 4032,
    "eval_val": True,
    "num_workers": 80,
    "apex": False,
    "debug": False,
    "scheduler_type": "constant_schedule_with_warmup",
    "n_warmup_steps": 1500,
    "cosine_n_cycles": 0.5,
    "learning_rate": 1.6e-4,
    "resume": f"../weights/{exp_id}.pt",
}

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
        a = torch.tile(inputs[:,:input_single_num].reshape(-1,1,input_single_num),(1,60,1))
        b = inputs[:,input_single_num:].reshape(-1,input_series_num,60).permute(0,2,1)
        inputs = torch.concat([a,b], dim=-1).squeeze(0)
        
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
    def __init__(self, inputs_dim=25, num_lstm=5, hidden_state=256, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.inputs_dim = inputs_dim
        self.output_single_num = 8

        residual_layers = nn.ModuleList()
        lstm_layers = nn.ModuleDict()
        self.mamba_layers = nn.ModuleList()
        # 定义LSTM层
        for i in range(num_lstm):  # 假设有5个LSTM层
            lstm_key = f'lstm{i + 1}'
            lstm_layers[lstm_key] = nn.LSTM(input_size=self.inputs_dim if i == 0 else hidden_state*2,
                                            hidden_size=hidden_state, num_layers=1,
                                            bidirectional=True, batch_first=True)
            self.mamba_layers.append(
                                    Mamba(
                                        # This module uses roughly 3 * expand * d_model^2 parameters
                                        d_model=hidden_state*2, # Model dimension d_model
                                        d_state=d_state,  # SSM state expansion factor
                                        d_conv=d_conv,    # Local convolution width
                                        expand=expand,    # Block expansion factor
                                    )
            )
            if i == 0:
                residual_layers.append(nn.Sequential(nn.LayerNorm([60, 512]),nn.GELU()))
            else:
                residual_layers.append(nn.LayerNorm([60, 512]))


        self.res_act = nn.GELU()
        self.lstm_stack = lstm_layers
        self.residual_stack = residual_layers
        self.fc = nn.Sequential(
            nn.Linear(512, 14),
            # nn.ReLU()  # 可选，FC之后的激活函数
        )

    def forward(self, inputs):
        outputs = inputs

        # 逐层通过LSTM并应用残差连接
        for i, lstm in enumerate(self.lstm_stack.values()):
            outputs, _ = lstm(outputs)
            outputs = self.mamba_layers[i](outputs)
            outputs = self.residual_stack[i](outputs)
            if i > 0:
                outputs = self.res_act(0.6*outputs+0.4*last_outputs)  # 残差连接
                last_outputs = outputs
            else:  
                if i == 0:
                    last_outputs = outputs 
                else:
                    print('Error!')
                    exit(0)
                    

        outputs = self.fc(outputs)  # b,60,14

        single_part = outputs[:,:,:self.output_single_num]
        single_part = torch.mean(single_part, axis=1) # b,8
        series_part = outputs[:,:,self.output_single_num:]
        series_part = series_part.permute(0,2,1).reshape(-1,360) # b,360

        outputs = torch.concat([single_part, series_part], axis=1)
        return outputs
    
def train_loop(train_loader, val_loader, model, optimizer,
               scheduler, criterion, score_func, gid):
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
            loss = criterion(outputs, labels, gid)

            avg_train.update(loss.item(), train_batch_size)

            loss.backward()
            if CFG.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            optimizer.step()
            scheduler.step(epoch + step / len(train_loader))
            optimizer.zero_grad(set_to_none=True)

            # loss.val 当前batch的loss, loss.avg是epoch里面的avg
            # print出来的结果
            if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
                print('Epoch: [{0}][{1}/{2}] '
                      'Elapsed {remain:s} '
                      'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                      'Grad: {grad_norm:.4f}  '
                      'LR: {lr:.8f}  '
                      .format(epoch, step, len(train_loader),
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
                avg_val_loss, oof_score, oof_pred, zero_list_tmp = valid_one_epoch(model, val_loader, epoch, criterion, score_func, gid)
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
                                       os.path.join(CFG.save_path, f'{CFG.exp_id}_g{gid}.pt'))
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
       'ptend_q0001_8', 'ptend_q0001_9', 'ptend_q0001_10','ptend_q0001_11',
       'ptend_q0002_0', 'ptend_q0002_1', 'ptend_q0002_2', 'ptend_q0002_3',
       'ptend_q0002_4', 'ptend_q0002_5', 'ptend_q0002_6', 'ptend_q0002_7',
       'ptend_q0002_8', 'ptend_q0002_9', 'ptend_q0002_10', 'ptend_q0002_11',
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
GIDX_DICT = {}
GIDX_DICT[0] = np.arange(8)
for i in range(6):
    GIDX_DICT[i+1] = 8+i*60+np.arange(60)


def negative_r2(y_true,y_pred, gid=None):
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
    if gid is None:
        for gid in range(7):
            idx = GIDX_DICT[gid]
            print(np.array(score_list)[idx])
            print(f'g{gid} r2:',np.mean(np.array(score_list)[idx]))
        return -np.mean(score_list), zero_list
    else:
        idx = GIDX_DICT[gid]
        print(np.array(score_list)[idx])
        return -np.mean(np.array(score_list)[idx]), []


class LEAPLoss(nn.Module):
    def __init__(self, w=1):
        super(LEAPLoss, self).__init__()
        self.w = w

    def forward(self, y_pred, y_true, gid=None):
        if gid is None: # 展示每一个分组的loss
            # loss = F.smooth_l1_loss(y_pred[:, :], y_true[:, :])
            # for i in range(6):
            #     pred_diff = y_pred[:, 8+60*i+1:8+60*(i+1)] - y_pred[:, 8+60*i:8+60*(i+1)-1]
            #     label_diff = y_true[:, 8+60*i+1:8+60*(i+1)] - y_true[:, 8+60*i:8+60*(i+1)-1]
            #     loss += F.smooth_l1_loss(pred_diff, label_diff)*self.w / 6

            # for gid in range(7): # 每一个分组的loss, 展示用
            #     idx = GIDX_DICT[gid]
            #     loss = F.smooth_l1_loss(y_pred[:, idx], y_true[:, idx])
            #     if gid > 0:
            #         i = gid - 1
            #         pred_diff = y_pred[:, 8+60*i+1:8+60*(i+1)] - y_pred[:, 8+60*i:8+60*(i+1)-1]
            #         label_diff = y_true[:, 8+60*i+1:8+60*(i+1)] - y_true[:, 8+60*i:8+60*(i+1)-1]
            #         loss += F.smooth_l1_loss(pred_diff, label_diff)*self.w
            #     print(f'g{gid} loss', loss.item())
             # 整体的loss就不计算了
            return torch.FloatTensor([-1])
        else: #计算分组的loss
            idx = GIDX_DICT[gid]
            loss = F.smooth_l1_loss(y_pred[:, idx], y_true[:, idx])
            if gid > 0:
                i = gid - 1
                pred_diff = y_pred[:, 8+60*i+1:8+60*(i+1)] - y_pred[:, 8+60*i:8+60*(i+1)-1]
                label_diff = y_true[:, 8+60*i+1:8+60*(i+1)] - y_true[:, 8+60*i:8+60*(i+1)-1]
                loss += F.smooth_l1_loss(pred_diff, label_diff)*self.w
        return loss

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

    for step, (inputs, labels) in tqdm(enumerate(dataloader), desc='valid one epoch', total=len(dataloader)):
        inputs= inputs.to(CFG.device, non_blocking=True)
        labels = labels.to(CFG.device, non_blocking=True)

        batch_size_val = labels.size(0)

        with torch.no_grad():
            out_puts = model(inputs)
            loss = criterion(out_puts, labels, gid)

        avg_eval.update(loss.item(), batch_size_val)

        oof_pred_list.append(out_puts.detach().cpu().numpy())
        oof_label_list.append(labels.detach().cpu().numpy())
    print()

    # loss.val是当前的metric loss.avg是历史平均
    print('EVAL: [{0}/{1}] '
          'Elapsed {remain:s} '
          'Loss: {loss.val:.4f}({loss.avg:.4f}) '
          .format(step+1, len(dataloader),
                  loss=avg_eval,
                  remain=timeSince(start, float(step + 1) / len(dataloader))))

    oof_pred = np.concatenate(oof_pred_list, axis=0)
    oof_label = np.concatenate(oof_label_list, axis=0)
    # 如果有多个打分，可以在这里处理下，最后logger相应的做处理即可
    for _ in zero_index:
        col_ = target_cols[_]
        oof_pred[:,_] = -mean_dict[col_]/std_dict[col_]

    oof_score, zero_list_tmp = score_func(oof_label, oof_pred, gid)
    time_consume = time.time() - start

    logger.info(f"epoch:{epoch},  "
                f"val_avg_loss:{avg_eval.avg:.4f},  "
                f"val_oof_score:{oof_score:.4f},  "
                f"time:{time_consume:.2f}")
    print(
        f"epoch:{epoch},  "
        f"val_avg_loss:{avg_eval.avg:.4f},  "
        f"val_oof_score:{oof_score:.4f},  "
        f"time:{time_consume:.2f}")
    print()
    del oof_label, oof_pred_list, oof_label_list, inputs
    torch.cuda.empty_cache()
    gc.collect()

    return avg_eval.avg, oof_score, oof_pred,zero_list_tmp

def pred_func_4g(inputs_array_path,model_dir,model_new,new_zero_pred_list):
    part1_input =np.load(inputs_array_path)
    test_dataset = LeapDataset(part1_input, mode='test')
    
    dataloader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=CFG.val_batch_size,
                                            num_workers=CFG.num_workers,
                                            shuffle=False, pin_memory=True, drop_last=False)

    oof_pred_llist=[]
    for gid in range(7):
        model_path = f"{model_dir}/{CFG.exp_id}_g{gid}.pt"
        print(f'load model from {model_path}')
        model = model_new
        model.load_state_dict(torch.load(model_path), strict=True)
        model = model.to(CFG.device)
        model.eval()
        oof_pred_list = []

        for step, inputs in tqdm(enumerate(dataloader), desc='test one epoch', total=len(dataloader)):
            inputs= inputs.to(CFG.device, non_blocking=True)

            with torch.no_grad():
                out_puts = model(inputs)
                out_puts = out_puts.to(torch.float64)
            oof_pred_list.append(out_puts.detach().cpu().numpy())

        oof_pred = np.concatenate(oof_pred_list, axis=0)[:,GIDX_DICT[gid]]
        oof_pred_llist.append(oof_pred)
    oof_pred = np.concatenate(oof_pred_llist, axis=1)
    
    sub_12_27 = pd.read_parquet("../data/sub_12_27.parquet")
    # 归一化回去
    final_np = np.zeros(oof_pred.shape, dtype=np.float64)
    sub_sample_old= pd.read_csv("../../raw_data/kaggle-data/sample_submission_old.csv", nrows=1)
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
            
    sub_sample = pd.read_csv("../../raw_data/kaggle-data/sample_submission.csv")
    columns_to_convert = [col for col in sub_sample.columns if col not in columns_to_keep]
    sub_sample[columns_to_convert] = sub_sample[columns_to_convert].astype(np.float64)
    sub_sample.iloc[:,1:] = final_np
    pl.from_pandas(sub_sample).write_parquet(f"../infer_outputs/{exp_id}/exp{exp_id}_new.parquet")

if __name__ == "__main__":

    tmp_model = LeapModel()
    tmp_model = nn.DataParallel(tmp_model.to(CFG.device), device_ids=list(range(3)))
    pred_func_4g(f"../data/test_0_inputs.npy",CFG.save_path,tmp_model,non_pred_list)