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
    "seed": 1103,
    "log_path": f"../outputs/{exp_id}/log.txt",
    "save_path": f"../outputs/{exp_id}",
    "device": "cuda",# cuda:0
    "print_freq": 1000,
    "save_best_model": True,
    # ======================== train & val ==========================
    "epoch": 1,
    "gradient_checkpointing_enable": False,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 1000,
    "evaluate_n_times_per_epoch": 4,
    "early_stop_flag": True,
    "early_stop_step": 0, # 1的话其实已经是两个epoch不增长了
    "batch_size": 1024*2,
    "val_batch_size": 2048*2,
    "eval_val": True,
    "num_workers": 24,
    "apex": False,
    "debug": False,
    "scheduler_type": "linear_schedule_with_warmup",
    "n_warmup_steps": 0,
    "cosine_n_cycles": 1,
    "learning_rate": 1e-3
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

# class LeapDataset(Dataset):
#     def __init__(self, inputs_array, outputs_array=None, mode = "train"):
        
#         super().__init__()
        
#         self.mode = mode
#         self.inputs = torch.from_numpy(inputs_array)
#         if self.mode == "train":
#             self.outputs = torch.from_numpy(outputs_array)
        
        
#     def __getitem__(self, idx):
#         inputs = self.inputs[idx]
#         inputs = inputs.to(torch.float32)

#         if self.mode == "train":
#             outputs = self.outputs[idx]
#             outputs = outputs.to(torch.float32)

#             return inputs, outputs
#         return inputs
    
#     def __len__(self):
#         return len(self.inputs)

class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)
    

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000, batch_first=False):
        super(PositionalEncoding, self).__init__()   
        self.batch_first = batch_first
        pe = torch.zeros(max_len, d_model) # seq,dim
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # seq,1
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))# dim
        pe[:, 0::2] = torch.sin(position * div_term) # seq,dim//2
        pe[:, 1::2] = torch.cos(position * div_term) # seq,dim//2
        pe = pe.unsqueeze(0).transpose(0, 1)# seq,1,dim
        #pe.requires_grad = False
        
        if self.batch_first:
            pe = pe.transpose(0,1)# 1,seq,dim
        self.register_buffer('pe', pe)

    def forward(self, x):
        if self.batch_first:
            return x + self.pe[:,:x.size(1),:x.size(2)]
        else:
            return x + self.pe[:x.size(0), :] # 支持变长序列
        
class LeapModel(nn.Module):
    def __init__(self, inputs_dim=25):
        super().__init__()
        self.inputs_dim = 25
        self.output_single_num = 8
        
        self.con1d2 = nn.Conv1d(inputs_dim,inputs_dim*3,kernel_size=2,padding='same')
        self.con1d3 = nn.Conv1d(inputs_dim,inputs_dim*3,kernel_size=4,padding='same')
        self.con1d4 = nn.Conv1d(inputs_dim,inputs_dim*3,kernel_size=6,padding='same')
        
        self.norm1 = nn.LayerNorm(self.inputs_dim*10)
        
        self.lstm1 = nn.LSTM(self.inputs_dim*10, 512, num_layers=6, bidirectional=True, batch_first=True)
        
        self.pos_encdoer = PositionalEncoding(26, max_len=60, batch_first=True)
        self.trans1 = nn.TransformerEncoderLayer(d_model = 1024, nhead = 16, dropout=0.,batch_first=True, dim_feedforward=2048,activation='gelu')
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024,256)
        self.fc3 = nn.Linear(256,14)
        #self.dropout = nn.Dropout(0.1)
        self._reinitialize()

    def forward(self, inputs, targets=None):
        
        inputs = self.pos_encdoer(inputs)
        
        inputs = inputs.permute(0,2,1)
        
        conv1d_output_ks2 =  F.relu(self.con1d2(inputs).permute(0,2,1))
        conv1d_output_ks3 =  F.relu(self.con1d3(inputs).permute(0,2,1))
        conv1d_output_ks4 =  F.relu(self.con1d4(inputs).permute(0,2,1))
        inputs = inputs.permute(0,2,1)
        
        inputs = torch.concat([inputs, conv1d_output_ks2, conv1d_output_ks3, conv1d_output_ks4],dim=-1)
        inputs = self.norm1(inputs)
        
        outputs,_ = self.lstm1(inputs) # b,60,200
        # outputs = self.pos_encdoer(outputs) # pos_encoder进入trans1之前
        outputs = torch.concat([outputs, self.trans1(outputs)], dim = -1)
        outputs = F.relu(self.fc1(outputs)) # b,60,14
        outputs = F.relu(self.fc2(outputs))
        outputs = self.fc3(outputs)
        
        single_part = outputs[:,:,:self.output_single_num]
        single_part = torch.mean(single_part, axis=1) # b,8
        series_part = outputs[:,:,self.output_single_num:]
        series_part = series_part.permute(0,2,1).reshape(-1,360) # b,360

        outputs = torch.concat([single_part, series_part], axis=1)
        return outputs
    
    # https://www.kaggle.com/code/junkoda/pytorch-lstm-with-tensorflow-like-initialization
    def _reinitialize(self):
        """
        Tensorflow/Keras-like initialization
        """
        for name, p in self.named_parameters():
            if 'lstm' in name:
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(p.data)
                elif 'bias_ih' in name:
                    p.data.fill_(0)
                    # Set forget-gate bias to 1
                    n = p.size(0)
                    p.data[(n // 4):(n // 2)].fill_(1)
                elif 'bias_hh' in name:
                    p.data.fill_(0)
            elif 'fc' in name:
                if 'weight' in name:
                    nn.init.xavier_uniform_(p.data)
                elif 'bias' in name:
                    p.data.fill_(0)

# class LeapModel(nn.Module):
#     def __init__(self, dims:list):
#         """
#         Initializes the LeapModel.

#         Parameters
#         ----------
#         dims : list of int
#             A list containing the dimensions of each layer in the network. 
#             The length of the list determines the number of layers.
#         """
        
#         super().__init__()
        
#         layers = []
#         for i in range(len(dims) - 2):
#             layers.append(nn.Linear(dims[i], dims[i + 1]))
#             layers.append(nn.LayerNorm(dims[i + 1]))
#             layers.append(nn.ReLU())
            
#         layers.append(nn.Linear(dims[-2], dims[-1]))
#         self.network = nn.Sequential(*layers)
        
#     def forward(self, x):
#         y = self.network(x)
#         return y
    
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
            for i in range(6):
                start_ahead = 8 + 60 * i + 1 # 9
                end_ahead = 8 + 60 * (i+1)
                start_behind = 8 + 60 * i
                end_behind = 8 + 60 * (i+1) - 1
                out_diff = outputs[:,start_ahead:end_ahead] - outputs[:,start_behind:end_behind]
                label_diff = labels[:,start_ahead:end_ahead] - labels[:,start_behind:end_behind]
                loss += criterion(out_diff, label_diff) / 6

            avg_train.update(loss.item(), train_batch_size)

            loss.backward()
            if CFG.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            optimizer.step()
            # scheduler.step()
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
                              lr=optimizer.param_groups[0]['lr']))

            # 多次打日志，一个epoch里train打印log
            if (step == (len(train_loader) - 1)) or \
                    (step + 1 in eval_steps):
                time_consume = time.time() - start

                logger.info(f"Epoch: [{epoch}][{step + 1}/{len(train_loader)}], "
                            f"train_avg_loss:{avg_train.avg:.4f}, "
                            f"grad norm2:{grad_norm:.2f}, "
                            f"time:{time_consume:.2f}, "
                            f"Lr:{optimizer.param_groups[0]['lr']:.8f}")

            
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
                scheduler.step(oof_score)
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

    
raw_data_path = "../../../raw_data"
adam_data_path = "../../data"
middle_result_path = os.path.join(adam_data_path,"middle_result")

# mean/std value for every cols
with open(os.path.join(middle_result_path, "mean.json"),'r') as f:
    mean_dict = json.load(f)

with open(os.path.join(middle_result_path, "std.json"),'r') as f:
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
       'ptend_q0002_12', 'ptend_q0002_13', 'ptend_q0002_14', 'ptend_q0003_0',
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

linear_corr_list = [f'ptend_q0002_{i}' for i in range(15,28)]

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
    if torch.cuda.device_count() > 1:
        # logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
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

    oof_pred = np.concatenate(oof_pred_list, axis=0)
    
    sub_15_27 = pd.read_parquet(os.path.join(adam_data_path, "adam_test_batch_new/sub_15_27.parquet"))
    # 归一化回去
    final_np = np.zeros(oof_pred.shape, dtype=np.float64)
    for idx, col in enumerate(sub_order):
        # 直接0
        if col in new_zero_pred_list:
            final_np[:,idx] = 0.0
            continue
        elif col in linear_corr_list:
            final_np[:,idx] = sub_15_27[col].values
        else: 
            old_idx = target_cols.index(col)
            final_np[:,idx] = (oof_pred[:,old_idx] * std_dict[col] + mean_dict[col]).astype(np.float64) /  old_weight[col]
            
    sub_sample = pd.read_csv(os.path.join(raw_data_path, "kaggle-data/sample_submission.csv"))
    sub_sample.iloc[:,1:] = final_np
    sub_sample.to_parquet(f"../outputs/{exp_id}/exp{exp_id}_new.parquet")


if __name__ == "__main__":
    part0_input = np.load(os.path.join(adam_data_path, "adam_full/sample4_inputs_v3.npy"))
    part0_output = np.load(os.path.join(adam_data_path, "adam_full/sample4_outputs_v3.npy"))

    part1_input = np.load(os.path.join(adam_data_path, "adam_full/new_test_8_9_inputs.npy"))
    part1_output = np.load(os.path.join(adam_data_path, "adam_full/new_test_8_9_outputs.npy"))
    
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


    model = LeapModel()
    if torch.cuda.device_count() > 1:
        logger.info(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    model = model.to(CFG.device)#LeapModel()
    tmp_model = LeapModel()


    optimizer = Adam(model.parameters(), lr=CFG.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True, min_lr=1e-5)
    criterion = torch.nn.SmoothL1Loss()#torch.nn.MSELoss()
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
    
    sub = pd.read_csv(os.path.join(middle_result_path, "old_sample_weight.csv"))
    old_weight = sub.iloc[0,:].to_dict()
    non_pred_list = ['ptend_q0001_0', 'ptend_q0001_1', 'ptend_q0001_2', 'ptend_q0001_3',
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
           'ptend_v_9', 'ptend_v_10', 'ptend_v_11'] 


    linear_corr_list = [f'ptend_q0002_{i}' for i in range(12,28)]

    pred_func(os.path.join(adam_data_path, "adam_test_batch_new/test_0_inputs.npy"),os.path.join(CFG.save_path, f'{CFG.exp_id}.pt'),tmp_model,non_pred_list)
    #logger.info(f"Mean folds score:{np.mean(overall_scores):.5f}")
    #logger.info(f"oof folds score:{score_func(train['score'],overall_oof):.5f}")