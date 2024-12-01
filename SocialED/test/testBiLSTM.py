import sys
sys.path.append('/home/zhangkun/py_projects/socialEDv3/SocialED/src/')

import pytest
import numpy as np
import pandas as pd
import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector.BiLSTM import *
from torch.utils.data import DataLoader

# 创建一个示例数据集
@pytest.fixture
def sample_dataset():
    data = {
        'event_id': [1, 2, 1, 2, 3, 1, 2, 3, 1, 2],  # 增加样本数量
        'words': [['hello', 'world'], ['goodbye', 'world'], ['hello', 'world'], ['goodbye', 'world'], ['new', 'event'],
                  ['hello', 'world'], ['goodbye', 'world'], ['new', 'event'], ['hello', 'world'], ['goodbye', 'world']],
        'filtered_words': [['hello', 'world'], ['goodbye', 'world'], ['hello', 'world'], ['goodbye', 'world'], ['new', 'event'],
                           ['hello', 'world'], ['goodbye', 'world'], ['new', 'event'], ['hello', 'world'], ['goodbye', 'world']]
    }
    return pd.DataFrame(data)

# 测试数据预处理
def test_preprocess(sample_dataset):
    bilstm = BiLSTM(sample_dataset)
    df = bilstm.preprocess()
    assert 'event_id' in df.columns
    assert 'words' in df.columns
    assert 'filtered_words' in df.columns
    assert 'wordsidx' in df.columns

# 测试数据分割
def test_split(sample_dataset):
    bilstm = BiLSTM(sample_dataset)
    bilstm.split()
    assert bilstm.train_df is not None
    assert bilstm.test_df is not None
    assert bilstm.val_df is not None

# 测试加载嵌入
def test_load_embeddings(sample_dataset):
    bilstm = BiLSTM(sample_dataset)
    bilstm.preprocess()
    bilstm.load_embeddings()
    assert bilstm.weight is not None
    assert bilstm.weight.shape[0] == len(bilstm.word2idx)

# 测试LSTM模型
def test_lstm_model(sample_dataset):
    bilstm = BiLSTM(sample_dataset)
    bilstm.preprocess()
    bilstm.load_embeddings()
    lstm_model = LSTM(bilstm.embedding_size, bilstm.weight, bilstm.num_hidden_nodes, bilstm.hidden_dim2,
                      bilstm.num_layers, bilstm.bi_directional, bilstm.dropout_keep_prob, bilstm.pad_index,
                      bilstm.batch_size)
    assert lstm_model is not None

# 测试VectorizeData类
def test_vectorize_data(sample_dataset):
    bilstm = BiLSTM(sample_dataset)
    bilstm.preprocess()
    vectorized_data = VectorizeData(bilstm.df, bilstm.max_len)
    assert vectorized_data is not None
    assert len(vectorized_data) == len(bilstm.df)

# 测试OnlineTripletLoss类
def test_online_triplet_loss(sample_dataset):
    bilstm = BiLSTM(sample_dataset)
    bilstm.preprocess()
    bilstm.load_embeddings()
    lstm_model = LSTM(bilstm.embedding_size, bilstm.weight, bilstm.num_hidden_nodes, bilstm.hidden_dim2,
                      bilstm.num_layers, bilstm.bi_directional, bilstm.dropout_keep_prob, bilstm.pad_index,
                      bilstm.batch_size)
    vectorized_data = VectorizeData(bilstm.df, bilstm.max_len)
    train_iterator = DataLoader(vectorized_data, batch_size=bilstm.batch_size, shuffle=True)
    loss_func = OnlineTripletLoss(bilstm.margin, RandomNegativeTripletSelector(bilstm.margin))
    for batch in train_iterator:
        text, text_lengths = batch['text']
        predictions = lstm_model(text, text_lengths)
        loss, num_triplets = loss_func(predictions, batch['label'])
        assert loss is not None
        assert num_triplets > 0

# 测试DatasetLoader类
def test_dataset_loader():
    loader = DatasetLoader(dataset='arabic_twitter')
    df = loader.load_data()
    assert df is not None
    assert isinstance(df, pd.DataFrame)

# 测试BiLSTM类的fit方法
def test_fit(sample_dataset):
    bilstm = BiLSTM(sample_dataset)
    bilstm.preprocess()
    bilstm.fit()
    assert bilstm.best_model is not None
    assert bilstm.best_epoch is not None

# 测试BiLSTM类的detection方法
def test_detection(sample_dataset):
    bilstm = BiLSTM(sample_dataset)
    bilstm.preprocess()
    bilstm.fit()
    ground_truths, predictions = bilstm.detection()
    assert ground_truths is not None
    assert predictions is not None
    assert len(ground_truths) == len(predictions)

# 测试BiLSTM类的evaluate方法
def test_evaluate(sample_dataset):
    bilstm = BiLSTM(sample_dataset)
    bilstm.preprocess()
    bilstm.fit()
    ground_truths, predictions = bilstm.detection()
    bilstm.evaluate(ground_truths, predictions)

if __name__ == "__main__":
    pytest.main()