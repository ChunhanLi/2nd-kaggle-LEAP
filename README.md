# LEAP - Atmospheric Physics using AI (ClimSim) - 2nd Place Solution

It's 2nd place solution to Kaggle competition: https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim

This repo contains the code we used to train the models. But it could be really time-consuming. As a result, we also provide trained model file to infer directly.



# How to train/infer

We have 5 people in our team and each one has his own environment and training/inference details.


1. [Common part](#Common-part)
2. [ADAM's part](#ADAM's-part)
3. [FDZM's part](#fdzm's-part)
    1.[Preprocessing](#preprocessing-part)
    2.[ForcewithMe's part](#forcewithmes-part)
    3.[Joseph's part](#josephs-partcoming-soon)
    4.[Max2020](#max2020s-part)
    5.[Zuiye](#zuiyes-part)
4. [Ensemble](#ensemble-part)
## Common part

- download [kaggle-data](https://www.kaggle.com/competitions/leap-atmospheric-physics-ai-climsim/data) into `raw_data/kaggle-data` folder. At least, we need those 3 files.
    - test.csv
    - sample_submission.csv
    - sample_submission_old.csv
- [**if you only plan to do the inference, this step can be skipped.**] download https://huggingface.co/datasets/LEAP/ClimSim_low-res data into `raw_data/ClimSim_low-res` folder. The expected structure should be `raw_data/ClimSim_low-res/train/0009-01/*.nc`

## ADAM's part

**HARDWARE**


- RAM: At least 360 Gi [**if only infer, 120Gi is enough**]
- GPU: 3 x RTX4090 [**if only infer, 1 RTX4090 is enough**]

**SOFTWARE**

- Python 3.8.10
    - adam_part/requirements.txt   
- CUDA 12.2
- nvidia drivers v535.129.03

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


## fdzm's part

### Preprocessing part
- 注意区分下train from scratch和only inference的preprocess部分。

### ForcewithMe's part 

### Joseph's part(coming soon)

### Max2020's part
My environment requirements need to be consistent with those of Joseph and Forcewithme.

#### Train from scratch

```shell
cd fdzm_part/exp
sh train_max2020.sh 
```
#### Only inference
```shell
cd fdzm_part/infer
sh infer_max2020.sh
```

In my section, I focused exclusively on the fine-tuning of the LSTM model. Model 10 follows the approach designed by [@zui0711](https://www.kaggle.com/zui0711). The architecture of Model 10 consists of two connected LSTM layers with different hidden sizes, followed by a MultiheadAttention layer. Models 14, 15, 21, and 22 are all improvements based on the model by [@forcewithme](https://www.kaggle.com/forcewithme), integrating LSTM with skip connections. Model 22 is our team’s highest-performing single model, providing us with the best results in local scoring, Leader Board scoring, and private scoring.

<div align=center><img src="https://github.com/user-attachments/assets/721c1783-69a1-4d93-9184-c3a52c69211c" alt="jpg name" width="50%"/></div>

Regarding the learning rate schedule, I used a cosine decay learning rate, with decays occurring at three and six epochs.

<div align=center><img src="https://github.com/user-attachments/assets/f21a9ffd-00e5-4a14-9337-d8937e5bf017" alt="jpg name" width="80%"/></div>

For the loss function, I utilized smooth L1 loss with a beta of 0.5.

#### Group Finetune
In deep learning, a continuously discussed topic within multi-objective learning tasks is the interaction between different learning objectives, specifically whether they promote or inhibit each other. In our experiments on the leap dataset, we found that in the early stages of training, seven different target groups promoted each other. However, towards the end of the training, these learning objectives began to interfere with each other, potentially due to complex semantic constraints. 

Inspired by the [top solution from the 2021 VPP competition](https://www.kaggle.com/competitions/ventilator-pressure-prediction/discussion/285320), we divided 368 features into seven groups, six of which are series of measurements of different metrics along the atmospheric column, and one group consists of eight unique single targets. After the training process with 364 full outputs was completed, we fine-tuned these groups again. This allowed each model with different architectures to achieve an improvement ranging from 0.0005 to 0.0015. Due to time and resource constraints, we only fine-tuned each group for one epoch.

### ZuiYe's part
According to the final hill climb result, my models are not taken into use in the final ensemble submission, so I just simply describe my method without codes.

My models are mainly based one two architectures. The first one consists of 2 LSTM layers followed by a MultiheadAttention layer. The other one consists of 3 parallel Convolutional layers with 3 different kernal sizes and next 2 LSTM layers followed by a MultiheadAttention layer just like the first architecture. My best single model gets LB 0.78696 / PB 0.78205 and ensemle of my own models (with hill climb) gets LB 0.79050/ PB 0.78614.

#### Auxiliary Loss
I design an auxiliary loss we call Diff Loss to help our models learn better. Almost all models of our teams benefit from this. For every group of targets with 60 vertical levels, we caculate the difference of the real values of level N with level N+1 and the difference of predicted values of level N with level N+1. The error of prediction difference to real difference is caculated with smoothl1 loss to describe the changes between two adjacent levels and then added to the main loss. The code is as follows.

```python
with torch.no_grad():
    out_puts = model(inputs)
    loss = criterion(out_puts, labels)
    for i in range(6):
        output_diff = out_puts[:, 8+60*i+1:8+60*(i+1)] - out_puts[:, 8+60*i:8+60*(i+1)-1]
        label_diff = labels[:, 8+60*i+1:8+60*(i+1)] - labels[:, 8+60*i:8+60*(i+1)-1]
        loss += criterion(output_diff, label_diff) / 6
```

## Ensemble part

Finally, We use [hill climb](https://www.kaggle.com/competitions/playground-series-s3e3/discussion/379690) to search blend weights.

`submission/blend/weight_df_dict_all_group_all_v10.pt` saves ensemble weight of each model.

**Weights of best model are following:**

|exp_id|weight|cv|public leaderborad|private leaderboard|
|:-:|:-:|:-:|:-:|:-:|
|forcewithme_exp32|0.166556|0.790|0.7865|0.78398|
|forcewithme_exp37|0.158625|0.7896|0.78618|0.78293|
|forcewithme_exp38|0.139194|0.7897|0.78719|0.78362|
|max_exp22|0.120125|0.7908|**0.78793**|**0.78434**|
|Jo_exp912|0.111971|0.78935|0.78562|0.78139|
|max_exp21|0.104738|0.7904|0.78752|0.78425|
|forcewithme_exp39|0.098977|0.789|0.78699|0.78257|
|max_exp14|0.093088|0.7905|0.78641|0.78214|
|max_exp10|0.092157|0.7888|0.78619|0.78213|
|forcewithme_exp40|0.082941|0.7885|0.7853|0.78261|
|max_exp015|0.052500|0.7905|0.78695|0.78244|
|adam_exp197|0.048994|0.7855|0.78269|0.777
|adam_exp200|-0.047132|0.7836|0.78010|0.77434|
|adam_exp195|-0.049875|0.78569|0.78334|0.77753|
|Jo_exp907|-0.083779|0.7855|0.78289|0.77873|
|forcewithme_exp18|-0.089079|0.7890|0.7863|0.78272


**Code**

```
cd submission/blend
python hill_climb_blend.py
```

This will generate `submission/blend/final_blend_v10.parquet` for final submission.

- cv:0.7955 
- public leaderborad: 0.79211
- private leaderboard: 0.78856

