# LEAP - Atmospheric Physics using AI (ClimSim) - 2nd Place Solution

It's 2nd place solution to Kaggle competition: https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim

This repo contains the code we used to train the models. But it could be really time-consuming. AS as result, we also provide trained model file to infer directly.



# How to train/infer

We have 5 people in our team and each one has his own environment and training/inference details.

## Common part

- download [kaggle-data](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data) into `raw_data/kaggle-data` folder. At least, we need those 3 files.
    - test.csv
    - sample_submission.csv
    - sample_submission_old.csv
- [**if you only plan to do the inference, this step can be skipped.**] download https://huggingface.co/datasets/LEAP/ClimSim_low-res data into `raw_data/ClimSim_low-res` folder. The expected structure should be raw_data/ClimSim_low-res/train/0009-01/*.nc

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

From [link](https://www.kaggle.com/datasets/hookman/leap-2nd-prize-models), Download file `v3_index.pt` into `adam_part/data/middle_result` folder
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

- **STEP3: copy submission for ensemble**

move submission files to `submission/subs` folder for final ensemble
```
cd adam_part/src
sh cp_train.sh
```

### Only inference

In this part, it will only do the inference using model file we uploaded. 
- **STEP1: preprocessing**

1. Download model file `195.pt`, `197.pt`, `200.pt` from [link](https://www.kaggle.com/datasets/hookman/leap-2nd-prize-models) into `adam_part/src/infer/saved_model` folder
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

- **STEP3: copy submission for ensemble**

move submission files to `submission/subs` folder for final ensemble
```
cd adam_part/src
sh cp_infer.sh
```

## xxxx's part

## xxxx's part

## Max2020's part
In my section, I focused exclusively on the fine-tuning of the LSTM model. Model 10 follows the approach designed by [@zui0711](https://www.kaggle.com/zui0711). The architecture of Model 10 consists of two connected LSTM layers with different hidden sizes, followed by a MultiheadAttention layer. Models 14, 15, 21, and 22 are all improvements based on the model by [@forcewithme](https://www.kaggle.com/forcewithme), integrating LSTM with skip connections. Model 22 is our team’s highest-performing single model, providing us with the best results in local scoring, Leader Board scoring, and private scoring.

Regarding the learning rate schedule, I used a cosine decay learning rate, with decays occurring at three and six epochs.

For the loss function, I utilized smooth L1 loss with a beta of 0.5.

## Group Finetune
In deep learning, a continuously discussed topic within multi-objective learning tasks is the interaction between different learning objectives, specifically whether they promote or inhibit each other. In our experiments on the leap dataset, we found that in the early stages of training, seven different target groups promoted each other. However, towards the end of the training, these learning objectives began to interfere with each other, potentially due to complex semantic constraints. 

Inspired by the [top solution from the 2021 VPP competition](https://www.kaggle.com/competitions/ventilator-pressure-prediction/discussion/285320), we divided 368 features into seven groups, six of which are series of measurements of different metrics along the atmospheric column, and one group consists of eight unique single targets. After the training process with 364 full outputs was completed, we fine-tuned these groups again. This allowed each model with different architectures to achieve an improvement ranging from 0.0005 to 0.0015. Due to time and resource constraints, we only fine-tuned each group for one epoch.
## Ensemble part

Finally, We use [hill climb](https://www.kaggle.com/competitions/playground-series-s3e3/discussion/379690) to search blend weights.

`submission/blend/weight_df_dict_all_group_all_v10.pt` saves weights of each model.

```
cd submission/blend
python hill_climb_blend.py
```

This will generate `submission/blend/final_blend_v10.parquet` for final submission.
