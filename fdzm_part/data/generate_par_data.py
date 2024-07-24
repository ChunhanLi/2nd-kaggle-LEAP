import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import os

only_inference = True

if __name__ == "__main__":
    raw_data_path = "../../raw_data/kaggle-data/"
    train_save_path = "./train_batch/"
    os.makedirs(train_save_path, exist_ok=True)
    test_save_path = "./test_batch/"
    os.makedirs(test_save_path, exist_ok=True)

    if not only_inference:
        # 取sub权重
        sub_all = pd.read_csv(raw_data_path + "sample_submission_old.csv", chunksize=10)
        for sub in sub_all:
            break
        sub_cols = list(sub.columns[1:])
        sub_npy = sub.iloc[:, 1:].values[0, :]

        # 权重为0的列
        zero_cols = [
            'ptend_q0001_0', 'ptend_q0001_1', 'ptend_q0001_2', 'ptend_q0001_3',
            'ptend_q0001_4', 'ptend_q0001_5', 'ptend_q0001_6', 'ptend_q0001_7',
            'ptend_q0001_8', 'ptend_q0001_9', 'ptend_q0001_10', 'ptend_q0001_11',
            'ptend_q0002_0', 'ptend_q0002_1', 'ptend_q0002_2', 'ptend_q0002_3',
            'ptend_q0002_4', 'ptend_q0002_5', 'ptend_q0002_6', 'ptend_q0002_7',
            'ptend_q0002_8', 'ptend_q0002_9', 'ptend_q0002_10', 'ptend_q0002_11',
            'ptend_q0003_0','ptend_q0003_1', 'ptend_q0003_2', 'ptend_q0003_3', 'ptend_q0003_4',
            'ptend_q0003_5', 'ptend_q0003_6', 'ptend_q0003_7', 'ptend_q0003_8',
            'ptend_q0003_9', 'ptend_q0003_10', 'ptend_q0003_11', 'ptend_u_0',
            'ptend_u_1', 'ptend_u_2', 'ptend_u_3', 'ptend_u_4', 'ptend_u_5',
            'ptend_u_6', 'ptend_u_7', 'ptend_u_8', 'ptend_u_9', 'ptend_u_10',
            'ptend_u_11', 'ptend_v_0', 'ptend_v_1', 'ptend_v_2', 'ptend_v_3',
            'ptend_v_4', 'ptend_v_5', 'ptend_v_6', 'ptend_v_7', 'ptend_v_8',
            'ptend_v_9', 'ptend_v_10', 'ptend_v_11'
        ]

        # train 文件生成
        train_iter = pd.read_csv(raw_data_path + "train.csv", chunksize=625000)
        k = 0
        for train_batch in tqdm(train_iter):
        #    # 乘以权重
            train_batch[sub_cols] = train_batch[sub_cols] * sub_npy
            train_col = list(train_batch.columns)[1:557]
            test_col = list(train_batch.columns)[557:]
            train_batch['sample_id'] = train_batch['sample_id'].map(lambda x:x.split("_")[1]).astype(np.int32)

            float64_cols = set(test_col + [f'state_q0002_{idx}' for idx in range(12, 28)]).difference(zero_cols)
            float32_cols = set(test_col + train_col).difference(float64_cols)

            for col32 in tqdm(float32_cols):
                train_batch[col32] = train_batch[col32].astype(np.float32)
            train_batch.to_parquet(train_save_path + f"{k}.parquet")
            k += 1

    # test 文件生成
    train_batch = pd.read_csv(raw_data_path + "test.csv")

    train_col = list(train_batch.columns)[1:557]
    train_batch['sample_id'] = train_batch['sample_id'].map(lambda x:x.split("_")[1]).astype(np.int32)

    float64_cols = set([f'state_q0002_{idx}' for idx in range(12, 28)])
    float32_cols = set(train_col).difference(float64_cols)
    
    for col32 in tqdm(float32_cols):
        train_batch[col32] = train_batch[col32].astype(np.float32)
    train_batch.to_parquet(test_save_path + "0.parquet")
    