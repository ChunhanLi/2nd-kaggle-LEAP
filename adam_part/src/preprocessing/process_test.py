from tqdm import tqdm

import numpy as np
import pandas as pd
import polars as pl
import os
import json

from tqdm.auto import tqdm
import gc

if __name__ == "__main__":
    raw_data_path = "../../../raw_data"
    adam_data_path = "../../data"
    middle_result_path = os.path.join(adam_data_path,"middle_result")
    # old sample weight
    sub = pd.read_csv(os.path.join(middle_result_path, "old_sample_weight.csv"))
    sub_cols = list(sub.columns)
    sub_npy = sub.values[0,:]

    # generate new test file
    k = 0
    train_batch = pd.read_csv(os.path.join(raw_data_path,'kaggle-data/test.csv'))

    # 乘以权重
    train_col = list(train_batch.columns)[1:557]
    train_batch['sample_id'] = train_batch['sample_id'].map(lambda x:x.split("_")[1]).astype(np.int32)

    float64_cols = set([f'state_q0002_{idx}' for idx in range(12,29)])
    float32_cols = set(train_col).difference(float64_cols)
    print(len(float32_cols))
    for col32 in tqdm(float32_cols):
        train_batch[col32] = train_batch[col32].astype(np.float32)
    train_batch.to_parquet(os.path.join(adam_data_path, f"adam_test_batch_new/{k}.parquet"))
    
    
    # normalization
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
    
    middle_result_path = os.path.join(adam_data_path,"middle_result")

    # mean/std value for every cols
    with open(os.path.join(middle_result_path, "mean.json"),'r') as f:
        mean_dict = json.load(f)

    with open(os.path.join(middle_result_path, "std.json"),'r') as f:
        std_dict = json.load(f)
        
        
    for num_ in [0]:
        print(f"====================={num_}===========================")
        train = pd.read_parquet(os.path.join(adam_data_path, f"adam_test_batch_new/{k}.parquet"))
        # inputs part
        mean_array = np.array([mean_dict[_] for _ in input_cols], dtype=np.float64)
        std_array = np.array([std_dict[_] for _ in input_cols], dtype=np.float64)
        inputs_array = train[input_cols].values
        inputs_array = ((inputs_array - mean_array) / std_array).astype(np.float32)
        np.save(os.path.join(adam_data_path, f"adam_test_batch_new/test_{num_}_inputs.npy"),inputs_array)
        
    # get linear relationship
    test = pd.read_parquet(os.path.join(adam_data_path, "adam_test_batch_new/0.parquet"))
    pred_15_27 = - test[[f'state_q0002_{i}' for i in range(12,29)]] / 1200
    pred_15_27.columns = [f'ptend_q0002_{i}' for i in range(12,29)]
    pred_15_27.to_parquet(os.path.join(adam_data_path, "adam_test_batch_new/sub_15_27.parquet"))
