#!/bin/bash

python3 test-model.py --name TF_EfficientNetV2S_21_Normalize0_Epochs300_Patience20_20240707_2104.h5 \
               --model 21 \
               --log_level 0 \
               --amount_image_test 3843 \
               --show_model

