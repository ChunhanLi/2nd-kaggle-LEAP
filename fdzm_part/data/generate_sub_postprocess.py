import pandas as pd

if __name__ == "__main__":
    test = pd.read_parquet("./test_batch/0.parquet")
    #sub_all = pd.read_csv("./sample_submission.csv", chunksize=10)
    #for sub in sub_all:
    #    break
    #weights = sub[[f'ptend_q0002_{i}' for i in range(12, 28)]].values[0, :]
    pred_12_27 = - test[[f'state_q0002_{i}' for i in range(12,28)]] / 1200  #* weights
    pred_12_27.columns = [f'ptend_q0002_{i}' for i in range(12,28)]
    pred_12_27.to_parquet("./sub_12_27.parquet")