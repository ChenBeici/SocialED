#!/bin/bash

# 定义模型列表及其对应的文件名
declare -A models
models=(
    ["LDA"]="LDA.py"
    ["BiLSTM"]="BiLSTM.py"
    ["WORD2VEC"]="word2vec.py"
    ["GloVe"]="GloVe.py"
    ["WMD"]="WMD.py"
    ["Bert"]="BERT.py"
    ["SBERT"]="SBERT.py"
    ["EventX"]="EventX.py"
    ["CLKD"]="CLKD.py"
    ["KPGNN"]="KPGNN.py"
    ["FinEvent"]="finevent.py"
    ["QSGNN"]="QSGNN.py"
    ["HCRC"]="HCRC.py"
    ["UCLSED"]="UCLSED.py"
    ["RPLM_SED"]="RPLMSED.py"
    ["HISEvent"]="HISEvent.py"
)

# 定义数据集
dataset="arabic_twitter"

# 遍历每个模型
for model in "${!models[@]}"; do
    # 获取模型对应的文件名
    model_file="${models[$model]}"
    
    # 创建输出文件夹
    mkdir -p "output/$model"
    
    # 使用 nohup 运行模型并将输出重定向到对应的文件
    nohup python "$model_file" > "output/$model/$dataset.txt" 2>&1 &
done


