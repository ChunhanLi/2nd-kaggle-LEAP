import pandas as pd
import json

train_save_path = "./train_batch/"
test_save_path = "./test_batch/"

if __name__ == "__main__":
    
    # 计算每列的mean/std
    tmp_df = pd.concat([pd.read_parquet(train_save_path + f"{i}.parquet") for i in range(15)], ignore_index=True)
    del tmp_df['sample_id']
    # singel col 
    tmp_mean_dict = tmp_df.mean(axis=0).to_dict()
    tmp_std_dict = tmp_df.std(axis=0).to_dict()

    mean_dict = {k: float(v) for k,v in tmp_mean_dict.items()}
    std_dict = {k: float(max(v, 1e-8)) for k,v in tmp_std_dict.items()}
    with open("./mean.json",'w') as f:
        json.dump(mean_dict, f)
    print("calc mean finished.")
    with open("./std.json",'w') as f:
        json.dump(std_dict, f)
    print("calc std finished.")
    