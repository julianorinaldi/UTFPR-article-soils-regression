#!/bin/bash

python3 main.py --name ResNet50[0]_Log[0]_Normalize[0]_Epochs[2]_Patience[1]_Separed[0]_ImgTrain[100]_ImgTest[50]_Preprocess[1] \
               --model 0 \
               --log_level 0 \
               --normalize 0 \
               --epochs 2 \
               --patience 1 \
               --amount_image_train 100 \
               --amount_image_test 50 \
               --preprocess
