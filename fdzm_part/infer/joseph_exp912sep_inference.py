import torch
import torch.nn as nn
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
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import time
from tqdm.auto import tqdm
import sys
sys.path.insert(0, "./")
import os
import gc

from joseph_model_utils import *
from joseph_base_utils import *

exp_id = "ex912sep"
config = {
    # ======================= global setting =====================
    "exp_id": "ex912",
    "seed": 315,
    "log_path": f"../infer_outputs/{exp_id}/log.txt",
    "save_path": f"../weights/{exp_id}",
    "device": "cuda:0",
    "save_best_model": True,
    # ======================== train & val ==========================
    "epoch": 9,
    "print_freq": 10000,
    "gradient_checkpointing_enable": False,
    "gradient_accumulation_steps": 1,
    "max_grad_norm": 4,
    "evaluate_n_times_per_epoch": 2,
    "early_stop_flag": True,
    "early_stop_step": 1000,
    "batch_size": 1024*3,
    "val_batch_size": 2048*3,
    "eval_val": True,
    "num_workers": 12,
    "apex": False,
    "scheduler_type": "CosineAnnealingWarmRestarts",
    "lr_lstm": 6e-4,
    "lr_gru": 6e-4,
    "lr_attention": 4e-4,
    "lr_fc": 8e-4
}

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

    
with open("../data/mean_v0.json",'r') as f:
    mean_dict = json.load(f)

with open("../data/std_v0.json",'r') as f:
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

zero_index = []
for _ in non_pred_list:
    zero_index.append(target_cols.index(_))

GIDX_DICT = {}
GIDX_DICT[0] = np.arange(8)
for i in range(6):
    GIDX_DICT[i+1] = 8+i*60+np.arange(60)


def pred_func(inputs_array_path, model_path, model_new, new_zero_pred_list):
    part1_input = np.load(inputs_array_path)
    test_dataset = LeapDataset(part1_input, mode='test')
    
    dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=CFG.val_batch_size,
        num_workers=CFG.num_workers,
        shuffle=False, pin_memory=True, drop_last=False
    )
    model = model_new
    model.load_state_dict(torch.load(model_path), strict=True)
    model = model.to(CFG.device)
    
    model.eval()
    oof_pred_list = []
    for step, inputs in enumerate(dataloader):
        inputs = inputs.to(CFG.device, non_blocking=True)

        with torch.no_grad():
            out_puts = model(inputs)
            out_puts = out_puts.to(torch.float64)
        oof_pred_list.append(out_puts.detach().cpu().numpy())

    oof_pred = np.concatenate(oof_pred_list, axis=0)
    sub_12_27 = pd.read_parquet("../data/sub_12_27.parquet")
    # reverse normalisation
    final_np = np.zeros(oof_pred.shape, dtype=np.float64)
    sub_sample_old = pd.read_csv("../../raw_data/kaggle-data/sample_submission_old.csv", nrows=1)
    for idx, col in enumerate(sub_order):
        # if a label is predicted poorly, just set it to 0.
        if col in new_zero_pred_list:
            final_np[:,idx] = 0.0
            continue
        elif col in linear_corr_list:
            final_np[:,idx] = sub_12_27[col].values
        else: 
            old_idx = target_cols.index(col)
            final_np[:,idx] = (oof_pred[:,old_idx] * std_dict[col] + mean_dict[col]) / sub_sample_old[col].values[0]
            
    sub_sample = pd.read_csv("../../raw_data/kaggle-data/sample_submission_new.csv")
    sub_sample.iloc[:,1:] = final_np
    sub_sample.to_parquet(f"../outputs/{exp_id}/{exp_id}.parquet")


def pred_func(inputs_array_path, model_dir, model_new, new_zero_pred_list):
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
    # reverse normalisation
    final_np = np.zeros(oof_pred.shape, dtype=np.float64)
    sub_sample_old = pd.read_csv("../../raw_data/kaggle-data/sample_submission_old.csv", nrows=1)
    for idx, col in enumerate(sub_order):
        # if a label is predicted poorly, just set it to 0.
        if col in new_zero_pred_list:
            final_np[:,idx] = 0.0
            continue
        elif col in linear_corr_list:
            final_np[:,idx] = sub_12_27[col].values
        else: 
            old_idx = target_cols.index(col)
            final_np[:,idx] = (oof_pred[:,old_idx] * std_dict[col] + mean_dict[col]) / sub_sample_old[col].values[0]
            
    sub_sample = pd.read_csv("../../raw_data/kaggle-data/sample_submission_new.csv")
    sub_sample.iloc[:,1:] = final_np
    sub_sample.to_parquet(f"../infer_outputs/{exp_id}/{exp_id}.parquet")


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    #model = LEAP_Model().to(CFG.device)
    #model = nn.DataParallel(model)
    tmp_model = LEAP_Model()
    pred_func(
        "../data/test_0_inputs.npy",
        os.path.join(CFG.save_path, f'{CFG.exp_id}.pt'),
        tmp_model,
        non_pred_list
    )


