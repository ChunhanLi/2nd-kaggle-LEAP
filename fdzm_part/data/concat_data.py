from tqdm.auto import tqdm
import numpy as np
import gc

def get_shuffle_list(l, ratio=0.5):
    nrange = np.arange(l)
    np.random.shuffle(nrange)
    idx = nrange[:int(l*ratio)]
    return idx

def get_list(l):
    idx = np.arange(l)
    return idx


def concat_npy_files_efficiently(npy_files):
    # 预先读取所有文件的形状和总大小
    shapes = [file.shape for file in npy_files]
    total_size = sum(shape[0] for shape in shapes)
    
    # 创建一个大的数组用于存储所有数据
    concatenated_array = np.empty((total_size,) + shapes[0][1:], dtype=np.float32)
    
    # 逐步加载每个文件的数据到预先分配的数组中
    current_position = 0
    for file, shape in tqdm(zip(npy_files, shapes)):
        concatenated_array[current_position:current_position + shape[0]] = file
        current_position += shape[0]
        gc.collect()
    
    return concatenated_array


def check_data_size(npy_files, ratio):
    # 预先读取所有文件的形状和总大小
    shapes = [np.load(file, mmap_mode='r').shape for file in npy_files]
    #idxs = [get_shuffle_list(shape[0], ratio=ratio) for i,shape in enumerate(shapes)]
    idxs = [get_list(shape[0]) for i,shape in enumerate(shapes)]
    total_size = sum(len(idx) for idx in idxs)
    print(total_size)


def load_npy_files_efficiently(npy_files, ratio, given_idxs=None):
    # 预先读取所有文件的形状和总大小
    shapes = [np.load(file, mmap_mode='r').shape for file in npy_files]
    if given_idxs:
        idxs = given_idxs
    else:
        if ratio < 1.0:
            idxs = [get_shuffle_list(shape[0], ratio=ratio) for i,shape in enumerate(shapes)]
        else:
            idxs = [get_list(shape[0]) for i,shape in enumerate(shapes)]
    total_size = sum(len(idx) for idx in idxs)
    print(total_size)
    
    # 创建一个大的数组用于存储所有数据
    concatenated_array = np.empty((total_size,) + shapes[0][1:], dtype=np.float32)
    
    # 逐步加载每个文件的数据到预先分配的数组中
    current_position = 0
    for file, idx in tqdm(zip(npy_files, idxs)):
        data = np.load(file, mmap_mode='r')
        data = data[idx, :]
        print(file, len(data))
        concatenated_array[current_position:current_position + len(idx)] = data
        current_position += len(idx)
        del data
        gc.collect()
    if given_idxs:
        return concatenated_array
    else:
        return concatenated_array, idxs
    

np.random.seed(315)
use_year_month = [
    '01_02', '01_03', '01_04', '01_05', '01_06', '01_07', '01_08', '01_09', '01_10', '01_11', '01_12',
    '02_01', '02_02', '02_03', '02_04', '02_05', '02_06', '02_07', '02_08', '02_09', '02_10', '02_11', '02_12', 
    '03_01', '03_02', '03_03', '03_04', '03_05', '03_06', '03_07', '03_08', '03_09', '03_10', '03_11', '03_12',
    '04_01', '04_02', '04_03', '04_04', '04_05', '04_06', '04_07', '04_08', '04_09', '04_10', '04_11', '04_12',
    '05_01', '05_02', '05_03', '05_04', '05_05', '05_06', '05_07', '05_08', '05_09', '05_10', '05_11', '05_12', 
    '06_01', '06_02', '06_03', '06_04', '06_05', '06_06', '06_07', '06_08', '06_09', '06_10', '06_11', '06_12',
    '07_01', '07_02', '07_03', '07_04', '07_05', '07_06', '07_07', '07_08', '07_09', '07_10', '07_11', '07_12', 
    '08_01', '08_02', '08_03', '08_04', '08_05', '08_06'
]

train_inputs_files = [f"./train_{ym}_inputs.npy" for ym in use_year_month]
train_outputs_files = [f"./train_{ym}_outputs.npy" for ym in use_year_month]


if __name__ == "__main__":
    full_inputs_array, idxs = load_npy_files_efficiently(train_inputs_files, ratio=1.0)
    np.save("./train_inputs_final.npy", full_inputs_array)
    del full_inputs_array
    for _ in range(4):
        gc.collect()

    full_outputs_array = load_npy_files_efficiently(train_outputs_files, ratio=1.0, given_idxs=idxs)
    np.save("./train_outputs_final.npy", full_outputs_array)
    del full_outputs_array
    for _ in range(4):
        gc.collect()