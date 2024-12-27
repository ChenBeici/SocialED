import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
from sklearn import metrics
from dataset.dataloader import DatasetLoader
from detector.wmd import *

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
    wmd = WMD(sample_dataset)
    df = wmd.preprocess()
    assert 'processed_text' in df.columns
    assert df['processed_text'].apply(lambda x: isinstance(x, list)).all()

# 测试训练 Word2Vec 模型
def test_fit(sample_dataset):
    wmd = WMD(sample_dataset)
    wmd.preprocess()
    wmd.fit()
    assert os.path.exists(wmd.model_path)

# 测试检测
def test_detection(sample_dataset):
    wmd = WMD(sample_dataset)
    wmd.preprocess()
    wmd.fit()
    ground_truths, predictions = wmd.detection()
    assert len(ground_truths) == len(predictions)

# 测试评估
def test_evaluate(sample_dataset):
    wmd = WMD(sample_dataset)
    wmd.preprocess()
    wmd.fit()
    ground_truths, predictions = wmd.detection()
    wmd.evaluate(ground_truths, predictions)

if __name__ == "__main__":
    pytest.main()