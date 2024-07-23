import numpy as np
import pandas as pd
import json

input_col_series_name = [
    "state_t", "state_q0001", "state_q0002", "state_q0003", "state_u", "state_v",
    "pbuf_ozone", "pbuf_CH4", "pbuf_N2O"
]
input_col_single =  [
    "state_ps","pbuf_SOLIN","pbuf_LHFLX","pbuf_SHFLX","pbuf_TAUX","pbuf_TAUY","pbuf_COSZRS",
    "cam_in_ALDIF","cam_in_ALDIR","cam_in_ASDIF","cam_in_ASDIR","cam_in_LWUP","cam_in_ICEFRAC",
    "cam_in_LANDFRAC","cam_in_OCNFRAC","cam_in_SNOWHLAND"
]
input_col_series = []
for _ in input_col_series_name:
    input_col_series += [ _ + f"_{i}" for i in range(60)]

input_cols = input_col_single + input_col_series


target_col_series_name = ["ptend_t", "ptend_q0001", "ptend_q0002", "ptend_q0003", "ptend_u", "ptend_v"]
target_col_single = [
    'cam_out_NETSW', "cam_out_FLWDS", "cam_out_PRECSC", "cam_out_PRECC",
    "cam_out_SOLS", "cam_out_SOLL", "cam_out_SOLSD", "cam_out_SOLLD"
]

target_col_series = []
for _ in target_col_series_name:
    target_col_series += [ _ + f"_{i}" for i in range(60)]
    
target_cols = target_col_single + target_col_series

with open("./mean.json", 'r') as f:
    mean_dict = json.load(f)

with open("./std.json", 'r') as f:
    std_dict = json.load(f)

if __name__ == "__main__":

    train = pd.read_parquet(f"./test_batch/0.parquet")
    # inputs part
    mean_array = np.array([mean_dict[_] for _ in input_cols], dtype=np.float32)
    std_array = np.array([std_dict[_] for _ in input_cols], dtype=np.float32)
    inputs_array = train[input_cols].values
    inputs_array = ((inputs_array - mean_array) / std_array).astype(np.float32)
    np.save(f"./test_0_inputs.npy", inputs_array)


    
