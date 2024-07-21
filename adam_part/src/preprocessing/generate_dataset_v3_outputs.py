import pandas as pd
import numpy as np
import glob
import gc
from tqdm.auto import tqdm
import torch
import os

np.random.seed(1103)

def get_shuffle_list(length,ratio=0.25):
    tt = np.arange(length)
    np.random.shuffle(tt)
    idx = tt[:int(ratio*length)]
    return idx



if __name__ == "__main__":
    raw_data_path = "../../raw_data"
    adam_data_path = "../../data"
    middle_result_path = os.path.join("../../data","middle_result")
    save_path = os.path.join(adam_data_path,"adam_full")
    
    # debug
    inputs_file_list_8 = [
         os.path.join(save_path,'train_08_01_inputs.npy'),
         os.path.join(save_path,'train_08_02_inputs.npy'),
         os.path.join(save_path,'train_08_03_inputs.npy'),
         os.path.join(save_path,'train_08_04_inputs.npy'),
         os.path.join(save_path,'train_08_05_inputs.npy'),
         os.path.join(save_path,'train_08_06_inputs.npy')
    ]
    inputs_file_list_7 = glob.glob(os.path.join(save_path, "train_07*inputs.npy"), recursive=True)
    inputs_file_list_6 = glob.glob(os.path.join(save_path, "train_06*inputs.npy"), recursive=True)
    inputs_file_list_5 = glob.glob(os.path.join(save_path, "train_05*inputs.npy"), recursive=True)
    inputs_file_list_4 = glob.glob(os.path.join(save_path, "train_04*inputs.npy"), recursive=True)
    inputs_file_list_3 = glob.glob(os.path.join(save_path, "train_03*inputs.npy"), recursive=True)
    inputs_file_list_2 = glob.glob(os.path.join(save_path, "train_02*inputs.npy"), recursive=True)
    inputs_file_list_1 = glob.glob(os.path.join(save_path, "train_01*inputs.npy"), recursive=True)

    inputs_file_list = inputs_file_list_1 + inputs_file_list_2 + inputs_file_list_3+inputs_file_list_4+inputs_file_list_5 + inputs_file_list_6 + inputs_file_list_7 + inputs_file_list_8

    array_dict = torch.load( os.path.join(middle_result_path, "v3_index.pt"))

    tmp_list_inputs = []
    tmp_list_outputs = []
    k = 0
    for input_file_ in tqdm(inputs_file_list):
        output_file_ = input_file_.replace("inputs","outputs")
        id_ = input_file_.split("../../data/adam_full/train_")[1].split("_inputs.np")[0]
        print(id_)
        print(input_file_,output_file_)
        print(k)
        #tmp_array_inputs = np.load(output_file_)
        #print(tmp_array_inputs.shape)
        #length_ = len(tmp_array_inputs)
        #idxs = get_shuffle_list(length_,ratio=0.65)
        idxs = array_dict[id_]

        #tmp_array_outputs = np.load(output_file_)[idxs,:]
        tmp_array_inputs = np.load(output_file_)[idxs,:]
        tmp_list_inputs.append(tmp_array_inputs)
        #tmp_list_outputs.append(tmp_array_outputs)
        del tmp_array_inputs#,tmp_array_outputs
        gc.collect()
        k+=1
    full_sample_inputs = np.concatenate(tmp_list_inputs,axis=0)
    print(len(tmp_list_inputs))
    print(full_sample_inputs.shape)
    np.save(os.path.join(save_path, "sample4_outputs_v3.npy"), full_sample_inputs)