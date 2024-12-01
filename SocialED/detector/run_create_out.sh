#!/bin/bash

# 创建 output 文件夹
mkdir -p output

# 定义模型列表
models=("LDA" "BiLSTM" "WORD2VEC" "GloVe" "WMD" "Bert" "SBERT" "EventX" "CLKD" "KPGNN" "FinEvent" "QSGNN" "HCRC" "UCLSED" "RPLM_SED" "HISEvent")

# 定义数据集列表
datasets=("maven" "event2012" "event2018" "arabic_twitter")

# 遍历每个模型
for model in "${models[@]}"; do
    # 在 output 文件夹中创建模型文件夹
    mkdir -p "output/$model"
    
    # 遍历每个数据集
    for dataset in "${datasets[@]}"; do
        # 在模型文件夹中创建数据集文件
        touch "output/$model/$dataset.txt"
    done
done