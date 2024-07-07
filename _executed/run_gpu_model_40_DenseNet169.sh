#!/bin/bash

python3 main.py --name DenseNet169_40_Normalize0_Epochs100_Patience10_Separed_Preprocess \
               --normalize 0 \
               --log_level 0 \
               --model 40 \
               --epochs 100 \
               --patience 10 \
               --preprocess

python3 main.py --name DenseNet169_40_Normalize1_Epochs100_Patience10_Separed_Preprocess \
               --normalize 1 \
               --log_level 0 \
               --model 40 \
               --epochs 100 \
               --patience 10 \
               --preprocess

python3 main.py --name DenseNet169_40_Normalize2_Epochs100_Patience10_Separed_Preprocess \
               --normalize 2 \
               --log_level 0 \
               --model 40 \
               --epochs 100 \
               --patience 10 \
               --preprocess

python3 main.py --name DenseNet169_40_Normalize3_Epochs100_Patience10_Separed_Preprocess \
               --normalize 3 \
               --log_level 0 \
               --model 40 \
               --epochs 100 \
               --patience 10 \
               --preprocess

python3 main.py --name DenseNet169_40_Normalize4_Epochs100_Patience10_Separed_Preprocess \
               --normalize 3 \
               --log_level 0 \
               --model 40 \
               --epochs 100 \
               --patience 10 \
               --preprocess

# Sem o preprocess
python3 main.py --name DenseNet169_40_Normalize0_Epochs100_Patience10_Separed_Preprocess \
               --normalize 0 \
               --log_level 0 \
               --model 40 \
               --epochs 100 \
               --patience 10 \

python3 main.py --name DenseNet169_40_Normalize1_Epochs100_Patience10_Separed_Preprocess \
               --normalize 1 \
               --log_level 0 \
               --model 40 \
               --epochs 100 \
               --patience 10 \

python3 main.py --name DenseNet169_40_Normalize2_Epochs100_Patience10_Separed_Preprocess \
               --normalize 2 \
               --log_level 0 \
               --model 40 \
               --epochs 100 \
               --patience 10 \

python3 main.py --name DenseNet169_40_Normalize3_Epochs100_Patience10_Separed_Preprocess \
               --normalize 3 \
               --log_level 0 \
               --model 40 \
               --epochs 100 \
               --patience 10 \

python3 main.py --name DenseNet169_40_Normalize4_Epochs100_Patience10_Separed_Preprocess \
               --normalize 3 \
               --log_level 0 \
               --model 40 \
               --epochs 100 \
               --patience 10 \