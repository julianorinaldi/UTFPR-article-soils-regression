#!/bin/bash

python main.py --name EfficientNetV2L_Normalize0_Epochs100_Patience10_Separed_Preprocess \
               --normalize 0 \
               --log_level 0 \
               --model 22 \
               --epochs 100 \
               --patience 10 \
               --preprocess

python main.py --name EfficientNetV2L_Normalize1_Epochs100_Patience10_Separed_Preprocess \
               --normalize 1 \
               --log_level 0 \
               --model 22 \
               --epochs 100 \
               --patience 10 \
               --preprocess

python main.py --name EfficientNetV2L_Normalize2_Epochs100_Patience10_Separed_Preprocess \
               --normalize 2 \
               --log_level 0 \
               --model 22 \
               --epochs 100 \
               --patience 10 \
               --preprocess

python main.py --name EfficientNetV2L_Normalize3_Epochs100_Patience10_Separed_Preprocess \
               --normalize 3 \
               --log_level 0 \
               --model 22 \
               --epochs 100 \
               --patience 10 \
               --preprocess

python main.py --name EfficientNetV2L_Normalize4_Epochs100_Patience10_Separed_Preprocess \
               --normalize 3 \
               --log_level 0 \
               --model 22 \
               --epochs 100 \
               --patience 10 \
               --preprocess

# Sem o preprocess
python main.py --name EfficientNetV2L_Normalize0_Epochs100_Patience10_Separed_Preprocess \
               --normalize 0 \
               --log_level 0 \
               --model 22 \
               --epochs 100 \
               --patience 10 \

python main.py --name EfficientNetV2L_Normalize1_Epochs100_Patience10_Separed_Preprocess \
               --normalize 1 \
               --log_level 0 \
               --model 22 \
               --epochs 100 \
               --patience 10 \

python main.py --name EfficientNetV2L_Normalize2_Epochs100_Patience10_Separed_Preprocess \
               --normalize 2 \
               --log_level 0 \
               --model 22 \
               --epochs 100 \
               --patience 10 \

python main.py --name EfficientNetV2L_Normalize3_Epochs100_Patience10_Separed_Preprocess \
               --normalize 3 \
               --log_level 0 \
               --model 22 \
               --epochs 100 \
               --patience 10 \

python main.py --name EfficientNetV2L_Normalize4_Epochs100_Patience10_Separed_Preprocess \
               --normalize 3 \
               --log_level 0 \
               --model 22 \
               --epochs 100 \
               --patience 10 \
