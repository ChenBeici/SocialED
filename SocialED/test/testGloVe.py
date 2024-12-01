import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from dataset.dataloader import DatasetLoader
from detector.GloVe import *

# 创建一个示例数据集
@pytest.fixture
def sample_dataset():
    data = {
        'event_id': [1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2],  # 增加样本数量
        'filtered_words': [
            ['hello', 'world'], ['goodbye', 'world'], ['hello', 'world'], ['goodbye', 'world'], ['new', 'event'],
            ['hello', 'world'], ['goodbye', 'world'], ['new', 'event'], ['hello', 'world'], ['goodbye', 'world'],
            ['hello', 'world'], ['goodbye', 'world'], ['new', 'event'], ['hello', 'world'], ['goodbye', 'world'],
            ['new', 'event'], ['hello', 'world'], ['goodbye', 'world'], ['new', 'event'], ['hello', 'world'],
            ['goodbye', 'world'], ['new', 'event'], ['hello', 'world'], ['goodbye', 'world'], ['new', 'event'],
            ['hello', 'world'], ['goodbye', 'world'], ['new', 'event'], ['hello', 'world'], ['goodbye', 'world']
        ]
    }
    return pd.DataFrame(data)

# 测试数据预处理
def test_preprocess(sample_dataset):
    glove = GloVe(sample_dataset)
    df = glove.preprocess()
    assert 'processed_text' in df.columns
    assert df['processed_text'].apply(lambda x: isinstance(x, list)).all()

# 测试加载 GloVe 向量
def test_load_glove_vectors():
    glove = GloVe(None)
    embeddings_index = glove.load_glove_vectors()
    assert isinstance(embeddings_index, dict)
    assert len(embeddings_index) > 0

# 测试文本转换为 GloVe 向量
def test_text_to_glove_vector(sample_dataset):
    glove = GloVe(sample_dataset)
    glove.load_glove_vectors()
    text = ['hello', 'world']
    vector = glove.text_to_glove_vector(text)
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (100,)

# 测试创建向量
def test_create_vectors(sample_dataset):
    glove = GloVe(sample_dataset)
    glove.load_glove_vectors()
    glove.preprocess()
    vectors = glove.create_vectors(glove.df, 'processed_text')
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == len(glove.df)
    assert vectors.shape[1] == 100

# 测试训练 KMeans 模型
def test_fit(sample_dataset):
    glove = GloVe(sample_dataset, num_clusters=5)  # 减少聚类数量
    glove.preprocess()
    glove.fit()
    assert os.path.exists(glove.model_path)

# 测试加载 KMeans 模型
def test_load_model(sample_dataset):
    glove = GloVe(sample_dataset, num_clusters=5)  # 减少聚类数量
    glove.preprocess()
    glove.fit()
    kmeans_model = glove.load_model()
    assert isinstance(kmeans_model, KMeans)

# 测试检测
def test_detection(sample_dataset):
    glove = GloVe(sample_dataset, num_clusters=5)  # 减少聚类数量
    glove.preprocess()
    glove.fit()
    ground_truths, predicted_labels = glove.detection()
    assert len(ground_truths) == len(predicted_labels)

# 测试评估
def test_evaluate(sample_dataset):
    glove = GloVe(sample_dataset, num_clusters=5)  # 减少聚类数量
    glove.preprocess()
    glove.fit()
    ground_truths, predicted_labels = glove.detection()
    glove.evaluate(ground_truths, predicted_labels)

if __name__ == "__main__":
    pytest.main()