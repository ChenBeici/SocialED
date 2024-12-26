import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn import metrics
from dataset.dataloader import DatasetLoader
from detector.sbert import *

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
    sbert = SBERT(sample_dataset)
    df = sbert.preprocess()
    assert 'processed_text' in df.columns
    assert df['processed_text'].apply(lambda x: isinstance(x, str)).all()

# 测试获取 SBERT 嵌入
def test_get_sbert_embeddings(sample_dataset):
    sbert = SBERT(sample_dataset)
    sbert.preprocess()
    text = 'hello world'
    embedding = sbert.get_sbert_embeddings(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (384,)  # 根据模型不同，维度可能不同

# 测试检测
def test_detection(sample_dataset):
    sbert = SBERT(sample_dataset)
    sbert.preprocess()
    ground_truths, predictions = sbert.detection()
    assert len(ground_truths) == len(predictions)

# 测试评估
def test_evaluate(sample_dataset):
    sbert = SBERT(sample_dataset)
    sbert.preprocess()
    ground_truths, predictions = sbert.detection()
    sbert.evaluate(ground_truths, predictions)

if __name__ == "__main__":
    pytest.main()