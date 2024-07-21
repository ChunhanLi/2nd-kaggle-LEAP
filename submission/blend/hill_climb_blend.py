# 模拟提交

import pandas as pd
import numpy as np
import os
import glob
import torch

target_col_series_name = ["ptend_t","ptend_q0001","ptend_q0002","ptend_q0003","ptend_u","ptend_v"]
target_col_single = ['cam_out_NETSW',"cam_out_FLWDS","cam_out_PRECSC","cam_out_PRECC","cam_out_SOLS",
                    "cam_out_SOLL","cam_out_SOLSD","cam_out_SOLLD"]

target_col_series = []
for _ in target_col_series_name:
    target_col_series += [ _ + f"_{i}" for i in range(60)]
    
target_cols = target_col_single + target_col_series

linear_corr_list = [f'ptend_q0002_{i}' for i in range(12,28)]

selected_idx = [i for i in range(len(target_cols)) if target_cols[i] not in linear_corr_list]
linear_idx = [i for i in range(len(target_cols)) if target_cols[i] in linear_corr_list]

if __name__ == "__main__":
    df = torch.load('./weight_df_dict_all_group_all_v10.pt')

    # ../subs
    base = pd.read_parquet("../subs/adam_195_78569.oof.parquet")

    submit_dict = {}
    for file in glob.glob("../subs/*.parquet", recursive=True):
        exp_id = file.split("/")[-1].split(".parquet")[0] +".npy"
        #print(exp_id)
        if exp_id in list(df['model']):
            print(exp_id)
            sub_ = pd.read_parquet(file)
            submit_dict[exp_id] = sub_

    # 验证
    tmp = 0
    w_sum = 0
    for model,weight in zip(df['model'],df['weight']):
        tmp += submit_dict[model].iloc[:,1:].values * weight
        w_sum +=weight
    base.iloc[:,1:] = tmp
    base.to_parquet("./final_blend_v10.parquet")
