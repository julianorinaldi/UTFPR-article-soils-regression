#!/bin/bash

python3 main.py --name EfficientNetB7_20_Normalize0_Epochs100_Patience10_Separed_Preprocess \
               --normalize 0 \
               --log_level 0 \
               --model 20 \
               --epochs 100 \
               --patience 10 \
               --preprocess

python3 main.py --name EfficientNetB7_20_Normalize1_Epochs100_Patience10_Separed_Preprocess \
               --normalize 1 \
               --log_level 0 \
               --model 20 \
               --epochs 100 \
               --patience 10 \
               --preprocess

python3 main.py --name EfficientNetB7_20_Normalize2_Epochs100_Patience10_Separed_Preprocess \
               --normalize 2 \
               --log_level 0 \
               --model 20 \
               --epochs 100 \
               --patience 10 \
               --preprocess

python3 main.py --name EfficientNetB7_20_Normalize3_Epochs100_Patience10_Separed_Preprocess \
               --normalize 3 \
               --log_level 0 \
               --model 20 \
               --epochs 100 \
               --patience 10 \
               --preprocess

python3 main.py --name EfficientNetB7_20_Normalize4_Epochs100_Patience10_Separed_Preprocess \
               --normalize 3 \
               --log_level 0 \
               --model 20 \
               --epochs 100 \
               --patience 10 \
               --preprocess

# Sem o preprocess
python3 main.py --name EfficientNetB7_20_Normalize0_Epochs100_Patience10_Separed \
               --normalize 0 \
               --log_level 0 \
               --model 20 \
               --epochs 100 \
               --patience 10 \

python3 main.py --name EfficientNetB7_20_Normalize1_Epochs100_Patience10_Separed \
               --normalize 1 \
               --log_level 0 \
               --model 20 \
               --epochs 100 \
               --patience 10 \

python3 main.py --name EfficientNetB7_20_Normalize2_Epochs100_Patience10_Separed \
               --normalize 2 \
               --log_level 0 \
               --model 20 \
               --epochs 100 \
               --patience 10 \

python3 main.py --name EfficientNetB7_20_Normalize3_Epochs100_Patience10_Separed \
               --normalize 3 \
               --log_level 0 \
               --model 20 \
               --epochs 100 \
               --patience 10 \

python3 main.py --name EfficientNetB7_20_Normalize4_Epochs100_Patience10_Separed \
               --normalize 3 \
               --log_level 0 \
               --model 20 \
               --epochs 100 \
               --patience 10 \
