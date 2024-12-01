import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from sklearn import metrics
from dataset.dataloader import DatasetLoader
from detector.BERT import *

# 创建一个示例数据集
@pytest.fixture
def sample_dataset():
    data = {
        'event_id': [1, 2, 1, 2, 3, 1, 2, 3, 1, 2],
        'filtered_words': [
            ['hello', 'world'], ['goodbye', 'world'], ['hello', 'world'], ['goodbye', 'world'], ['new', 'event'],
            ['hello', 'world'], ['goodbye', 'world'], ['new', 'event'], ['hello', 'world'], ['goodbye', 'world']
        ]
    }
    return pd.DataFrame(data)

# 测试数据预处理
def test_preprocess(sample_dataset):
    bert = BERT(sample_dataset)
    df = bert.preprocess()
    assert 'processed_text' in df.columns
    assert df['processed_text'].apply(lambda x: isinstance(x, str)).all()

# 测试获取 BERT 嵌入
def test_get_bert_embeddings(sample_dataset):
    bert = BERT(sample_dataset)
    bert.preprocess()
    text = 'hello world'
    embedding = bert.get_bert_embeddings(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)

# 测试检测
def test_detection(sample_dataset):
    bert = BERT(sample_dataset)
    bert.preprocess()
    ground_truths, predictions = bert.detection()
    assert len(ground_truths) == len(predictions)

# 测试评估
def test_evaluate(sample_dataset):
    bert = BERT(sample_dataset)
    bert.preprocess()
    ground_truths, predictions = bert.detection()
    ari, ami, nmi = bert.evaluate(ground_truths, predictions)
    assert isinstance(ari, float)
    assert isinstance(ami, float)
    assert isinstance(nmi, float)

if __name__ == "__main__":
    pytest.main()