import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from sklearn import metrics
from dataset.dataloader import DatasetLoader
from detector.eventx import *

# 创建一个示例数据集
@pytest.fixture
def sample_dataset():
    data = {
        'event_id': [1, 2, 1, 2, 3, 1, 2, 3, 1, 2],
        'filtered_words': [
            ['hello', 'world'], ['goodbye', 'world'], ['hello', 'world'], ['goodbye', 'world'], ['new', 'event'],
            ['hello', 'world'], ['goodbye', 'world'], ['new', 'event'], ['hello', 'world'], ['goodbye', 'world']
        ],
        'message_ids': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'entities': [
            [('hello', 'world')], [('goodbye', 'world')], [('hello', 'world')], [('goodbye', 'world')], [('new', 'event')],
            [('hello', 'world')], [('goodbye', 'world')], [('new', 'event')], [('hello', 'world')], [('goodbye', 'world')]
        ]
    }
    return pd.DataFrame(data)

# 测试数据预处理
def test_preprocess(sample_dataset):
    eventx = EventX(sample_dataset)
    eventx.preprocess()
    assert 'filtered_words' in eventx.df.columns
    assert 'message_ids' in eventx.df.columns
    assert 'entities' in eventx.df.columns

# 测试数据分割
def test_split(sample_dataset):
    eventx = EventX(sample_dataset)
    eventx.split()
    assert eventx.train_df is not None
    assert eventx.test_df is not None
    assert eventx.val_df is not None

# 测试事件检测
def test_detection(sample_dataset):
    eventx = EventX(sample_dataset)
    eventx.preprocess()
    ground_truths, predictions = eventx.detection()
    assert len(ground_truths) == len(predictions)

# 测试评估
def test_evaluate(sample_dataset):
    eventx = EventX(sample_dataset)
    eventx.preprocess()
    ground_truths, predictions = eventx.detection()
    eventx.evaluate(ground_truths, predictions)

if __name__ == "__main__":
    pytest.main()