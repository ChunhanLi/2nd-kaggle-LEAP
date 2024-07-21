# Get started

1. `install -r requirements.txt`
2. place all the downloaded weights in `fdzm_part/weights` folder. 
For example, for the `forcewithme_gf_reslstm_cv0.790_lb0.785` model, place `forcewithme_gf_reslstm_cv0.790_lb0.785.pt` in `fdzm_part/weights/forcewithme_gf_reslstm_cv0.790_lb0.785/forcewithme_gf_reslstm_cv0.790_lb0.785.pt`

# Training(Optinal)
1. We have provided the checkpoints trained during the LEAP competition. So if you don't want to reproduce the training process, you can skip this part and focuse on inference part. If you want to re-train the models, please attach to the following steps:
2. `cd exp`
3. `bash train_force`. This scripts will train all the 6 models of ForcewithMe. The training contains 2 stages:

(1) Optimize on all of the 368 targets
(2) Resume on the weights produced on stage (1), and fine-tuning on 7 groups(60-60-60-60-60-8), respectively.

4. The training outputs are in `fdzm_part/outputs`, including checkpoints, oof, prediction files and logs. 

# Inference
1. `cd infer`
2. `bash infer_force.sh`. This scripts will infer all the 6 models trained by ForcewithMe.
3. The inference outputs(parquet prediction files) are in `fdzm_part/outputs`.