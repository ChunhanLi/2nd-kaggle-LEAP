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
    "seed": 47,
    "log_path": f"../outputs/{exp_id}/log.txt",
    "save_path": f"../outputs/{exp_id}",
    "device": "cuda",# cuda:0
    "print_freq": 1000,
    "save_best_model": True,
    # ======================== train & val ==========================
    "epoch": 25,
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
seed_everything(seed=CFG.seed)

input_single_num = 16
input_series_num = 9
output_single_num = 8
output_series_num = 6


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
        
        self.con1d2 = nn.Conv1d(inputs_dim,inputs_dim*2,kernel_size=2,padding='same')
        self.con1d3 = nn.Conv1d(inputs_dim,inputs_dim*3,kernel_size=4,padding='same')
        self.con1d4 = nn.Conv1d(inputs_dim,inputs_dim*2,kernel_size=6,padding='same')
        
        self.norm1 = nn.LayerNorm(self.inputs_dim*8)
        
        self.lstm1 = nn.LSTM(self.inputs_dim*8, 256, num_layers=5, bidirectional=True, batch_first=True)
        
        self.pos_encdoer = PositionalEncoding(26, max_len=60, batch_first=True)
        self.trans1 = nn.TransformerEncoderLayer(d_model = 512, nhead = 8, dropout=0.,batch_first=True, dim_feedforward=1024)
        self.fc1 = nn.Linear(1024+200, 256)
        self.fc2 = nn.Linear(256,14)
        self.dropout = nn.Dropout(0.1)
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
        outputs = torch.concat([outputs, self.trans1(outputs), inputs], dim = -1)
        outputs = self.dropout(outputs)
        outputs = F.relu(self.fc1(outputs)) # b,60,14
        outputs = self.fc2(outputs)
        
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


def pred_func(inputs_array_path,model_path,model_new,new_zero_pred_list):
    part1_input =np.load(inputs_array_path)
    test_dataset = LeapDataset(part1_input, mode='test')
    
    dataloader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=CFG.val_batch_size,
                                            num_workers=CFG.num_workers,
                                            shuffle=False, pin_memory=True, drop_last=False)
    model = model_new

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
    sub_sample.to_parquet(f"./subs/exp{exp_id}_new.parquet")


if __name__ == "__main__":

    tmp_model = LeapModel()
    
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

    pred_func(os.path.join(adam_data_path, "adam_test_batch_new/test_0_inputs.npy"),os.path.join("./saved_model", f'{CFG.exp_id}.pt'),tmp_model,non_pred_list)
