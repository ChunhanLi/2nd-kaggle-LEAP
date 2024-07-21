from ClimSim.climsim_utils.data_utils import *

# set variable names

v2_inputs = ['state_t',
             'state_q0001',
             'state_q0002',
             'state_q0003',
             'state_u',
             'state_v',
             'state_ps',
             'pbuf_SOLIN',
             'pbuf_LHFLX',
             'pbuf_SHFLX',
             'pbuf_TAUX',
             'pbuf_TAUY',
             'pbuf_COSZRS',
             'cam_in_ALDIF',
             'cam_in_ALDIR',
             'cam_in_ASDIF',
             'cam_in_ASDIR',
             'cam_in_LWUP',
             'cam_in_ICEFRAC',
             'cam_in_LANDFRAC',
             'cam_in_OCNFRAC',
             'cam_in_SNOWHICE',
             'cam_in_SNOWHLAND',
             'pbuf_ozone', # outside of the upper troposphere lower stratosphere (UTLS, corresponding to indices 5-21), variance in minimal for these last 3 
             'pbuf_CH4',
             'pbuf_N2O']

v2_outputs = ['ptend_t',
              'ptend_q0001',
              'ptend_q0002',
              'ptend_q0003',
              'ptend_u',
              'ptend_v',
              'cam_out_NETSW',
              'cam_out_FLWDS',
              'cam_out_PRECSC',
              'cam_out_PRECC',
              'cam_out_SOLS',
              'cam_out_SOLL',
              'cam_out_SOLSD',
              'cam_out_SOLLD']

vertically_resolved = ['state_t', 
                       'state_q0001', 
                       'state_q0002', 
                       'state_q0003', 
                       'state_u', 
                       'state_v', 
                       'pbuf_ozone', 
                       'pbuf_CH4', 
                       'pbuf_N2O', 
                       'ptend_t', 
                       'ptend_q0001', 
                       'ptend_q0002', 
                       'ptend_q0003', 
                       'ptend_u', 
                       'ptend_v']

ablated_vars = ['ptend_q0001',
                'ptend_q0002',
                'ptend_q0003',
                'ptend_u',
                'ptend_v']

v2_vars = v2_inputs + v2_outputs

train_col_names = []
ablated_col_names = []
for var in v2_vars:
    if var in vertically_resolved:
        for i in range(60):
            train_col_names.append(var + '_' + str(i))
            if i < 12 and var in ablated_vars:
                ablated_col_names.append(var + '_' + str(i))
    else:
        train_col_names.append(var)

input_col_names = []
for var in v2_inputs:
    if var in vertically_resolved:
        for i in range(60):
            input_col_names.append(var + '_' + str(i))
    else:
        input_col_names.append(var)

output_col_names = []
for var in v2_outputs:
    if var in vertically_resolved:
        for i in range(60):
            output_col_names.append(var + '_' + str(i))
    else:
        output_col_names.append(var)

assert(len(train_col_names) == 17 + 60*9 + 60*6 + 8)
assert(len(input_col_names) == 17 + 60*9)
assert(len(output_col_names) == 60*6 + 8)
assert(len(set(output_col_names).intersection(set(ablated_col_names))) == len(ablated_col_names))

# initialize data_utils object

grid_path = 'ClimSim/grid_info/ClimSim_low-res_grid-info.nc'
norm_path = 'ClimSim/preprocessing/normalizations/'

grid_info = xr.open_dataset(grid_path)
input_mean = None#xr.open_dataset(norm_path + 'inputs/input_mean.nc')
input_max = None#xr.open_dataset(norm_path + 'inputs/input_max.nc')
input_min = None#xr.open_dataset(norm_path + 'inputs/input_min.nc')
output_scale = None#xr.open_dataset(norm_path + 'outputs/output_scale.nc')



from tqdm import tqdm
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from sklearn.metrics import r2_score as r2_score_sklearn
import logging
from tqdm.auto import tqdm
import gc
import json


input_col_series_name = ["state_t", "state_q0001", "state_q0002", "state_q0003", "state_u", "state_v","pbuf_ozone",
                   "pbuf_CH4","pbuf_N2O"]
input_col_single =  ['state_ps',"pbuf_SOLIN","pbuf_LHFLX","pbuf_SHFLX","pbuf_TAUX","pbuf_TAUY","pbuf_COSZRS",
                     "cam_in_ALDIF","cam_in_ALDIR","cam_in_ASDIF","cam_in_ASDIR","cam_in_LWUP","cam_in_ICEFRAC",
                     "cam_in_LANDFRAC","cam_in_OCNFRAC","cam_in_SNOWHLAND"]
input_col_series = []
for _ in input_col_series_name:
    input_col_series += [ _ + f"_{i}" for i in range(60)]

input_cols = input_col_single + input_col_series


target_col_series_name = ["ptend_t","ptend_q0001","ptend_q0002","ptend_q0003","ptend_u","ptend_v"]
target_col_single = ['cam_out_NETSW',"cam_out_FLWDS","cam_out_PRECSC","cam_out_PRECC","cam_out_SOLS",
                    "cam_out_SOLL","cam_out_SOLSD","cam_out_SOLLD"]

target_col_series = []
for _ in target_col_series_name:
    target_col_series += [ _ + f"_{i}" for i in range(60)]
    
target_cols = target_col_single + target_col_series


if __name__ == "__main__":
    raw_data_path = "../../../raw_data"
    adam_data_path = "../../data"
    middle_result_path = os.path.join(adam_data_path,"middle_result")
    # old sample weight
    sub = pd.read_csv(os.path.join(middle_result_path, "old_sample_weight.csv"))
    sub_cols = list(sub.columns)
    sub_npy = sub.values[0,:]

    # mean/std value for every cols
    with open(os.path.join(middle_result_path, "mean.json"),'r') as f:
        mean_dict = json.load(f)

    with open(os.path.join(middle_result_path, "std.json"),'r') as f:
        std_dict = json.load(f)



        
    data = data_utils(grid_info = grid_info, 
                      input_mean = input_mean, 
                      input_max = input_max, 
                      input_min = input_min, 
                      output_scale = output_scale)

    data.set_to_v2_vars()

    # do not normalize
    data.normalize = False

    # create training data

    # set data path for training data
    data.data_path = os.path.join(raw_data_path, 'ClimSim_low-res/train/')

    # set regular expressions for selecting training data
    # use last 7 months to valid 
    data.set_regexps(data_split = 'train', 
                    regexps = [f"E3SM-MMF.mli.0008-{_}-*-*.nc" for _ in ['07','08','09','10','11','12']] + [f'E3SM-MMF.mli.0009-01-*-*.nc']) 
    # # debug
    # data.set_regexps(data_split = 'train', 
    #                   regexps = [f'E3SM-MMF.mli.0009-01-*-*.nc']) 

    # set temporal subsampling
    data.set_stride_sample(data_split = 'train', stride_sample = 7)

    # create list of files to extract data from
    data.set_filelist(data_split = 'train')

    # save numpy files of training data
    data_loader = data.load_ncdata_with_generator(data_split = 'train')
    npy_iterator = list(data_loader.as_numpy_iterator())
    npy_input = np.concatenate([npy_iterator[x][0] for x in range(len(npy_iterator))])
    npy_output = np.concatenate([npy_iterator[x][1] for x in range(len(npy_iterator))])
    train_npy = np.concatenate([npy_input, npy_output], axis = 1)
    train_index = ["train_" + str(x) for x in range(train_npy.shape[0])]

    train = pd.DataFrame(train_npy, index = train_index, columns = train_col_names)
    train.index.name = 'sample_id'
    print('dropping cam_in_SNOWHICE because of strange values')
    train.drop('cam_in_SNOWHICE', axis=1, inplace=True)
    train = train.reset_index()
    train.to_parquet(os.path.join(adam_data_path, f"adam_full/test_val_08_09.parquet"))

    train[sub_cols] = train[sub_cols] * sub_npy
    train_col = list(train.columns)[1:557]
    test_col = list(train.columns)[557:]
    train['sample_id'] = train['sample_id'].map(lambda x:x.split("_")[1]).astype(np.int32)

    print(1)
    mean_array = np.array([mean_dict[_] for _ in input_cols], dtype=np.float64)
    std_array = np.array([std_dict[_] for _ in input_cols], dtype=np.float64)
    inputs_array = ((train[input_cols].values - mean_array) / std_array).astype(np.float32)
    np.save(os.path.join(adam_data_path ,f"adam_full/new_test_8_9_inputs.npy"),inputs_array)
    print(2)
    # output part
    mean_array = np.array([mean_dict[_] for _ in target_cols], dtype=np.float64)
    std_array = np.array([std_dict[_] for _ in target_cols], dtype=np.float64)  
    output_array = ((train[target_cols].values - mean_array) / std_array).astype(np.float32)
    np.save(os.path.join(adam_data_path, f"adam_full/new_test_8_9_outputs.npy"),output_array)



        
