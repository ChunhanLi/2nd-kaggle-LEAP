## Model 1：LSTM with residual Connection
python forcewithme_reslstm_cv0.789_lb0.783.py
python forcewithme_gf_reslstm_cv0.790_lb0.785.py

## Model 2：LSTM with residual Connection, large kernel 1dcnn encoder and lstm encoder
python forcewithme_cnnlstm_cv0.7881_lb0.784.py
python forcewithme_gf_cnnlstm_cv0.789_lb0.787.py

## Model 3：LSTM with residual Connection, stack several MAMBA layers as output layers
python forcewithme_lstmmamba_stage1.py
python forcewithme_lstmmamba_stage2.py
python forcewithme_gf_lstmmamba_cv0.7885_lb0.7853.py

## Model 4: Stack 2*LSTM+1*MANBA for 3 times
python forcewithme_LstmMambaMixed_cv0.7884_lb0.7855.py
python forcewithme_gf_LstmMambaMixed_cv0.7886_lb0.7858.py

# Model 5: Same model architecter as Model 4, only change the `d_state` to 64
python forcewithme_2LSTM1mamba-3_state64_cv0.7893_LB7868.py
python forcewithme_gf038_2LSTM1mamba-3_state64_cv0.7897_LB0.787.py

## Model 6: Stack 1*LSTM+1*MANBA for 5 times
python forcewithme_1LSTM1mamba-5_state16_cv0.7886_LB0.7858.py
python forcewithme_gf037_1LSTM1mamba-5_state16_cv0.7896_LBunknown.py