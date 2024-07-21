[toc]

# LEAP - Atmospheric Physics using AI (ClimSim) - 2nd Place Solution

It's 2nd place solution to Kaggle competition: https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim

This repo contains the code we used to train the models. But it could be really time-consuming. AS as result, we also provide trained model file to infer directly.



# How to train

We have 5 people in our team and each one has his own environments and training details.

## Common part

- download [kaggle-data](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data) into `raw_data/kaggle-data` folder. At least, we need those 3 files.
    - test.csv
    - sample_submission.csv
    - sample_submission_old.csv
- [**if you plan to train from scratch, this step is needed.**] download https://huggingface.co/datasets/LEAP/ClimSim_low-res data into `raw_data/ClimSim_low-res` folder. The expected structure should be raw_data/ClimSim_low-res/train/0009-01/*.nc

## ADAM's part

**HARDWARE**


- RAM: At least 360 Gi [**if only infer, 120Gi is enough**]
- GPU: 3 x RTX4090 [**if only infer, 1 RTX4090 is enough**]

**SOFTWARE**

- Python 3.8.10
    - adam_part/requirements.txt   
- CUDA 12.2
- nvidia drivers v535.129.03

### Train from scratch
- **STEP1: preprocessing**

Download model file `v3_index.pt` into `adam_part/data/middle_result` folder
- I used sampling in creating datasets. `v3_index.pt` is the sampling index which will be used when creating dataset.
```
cd adam_part/src
sh run_preprocess.sh 
```
- **STEP2: training**

In this part, it will train from scratch and also do the inference. The outputs are in adam_part/src/outputs folder. 
- oof.npy: valid oof 
- log.txt: log
- log_old.txt: the log when I trained
- exp{xxx}_new.parquet: submit file

```
cd adam_part/src/exp
sh run.sh
```

### Only inference

In this part, it will only do the inference using model file we uploaded. 
- **STEP1: preprocessing**

1. Download model file `195.pt`, `197.pt`, `200.pt` into `adam_part/src/infer/saved_model` folder
2. 
```
cd adam_part/src/preprocessing
python process_test.py
```
- **STEP2: inference**

The outputs are in adam_part/src/infer/subs folder. 
```
cd adam_part/src/infer
sh run_only_infer.sh
```
