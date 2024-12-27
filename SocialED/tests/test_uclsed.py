import pytest
import torch
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector.uclsed import *
# 创建一个虚拟的DatasetLoader实例
class MockDatasetLoader:
    def load_data(self):
        return pd.DataFrame({
            'tweet_id': [1, 2, 3],
            'user_mentions': [['user1'], ['user2'], ['user3']],
            'text': ['text1', 'text2', 'text3'],
            'hashtags': [['hashtag1'], ['hashtag2'], ['hashtag3']],
            'entities': [['entity1'], ['entity2'], ['entity3']],
            'urls': [['url1'], ['url2'], ['url3']],
            'filtered_words': [['word1'], ['word2'], ['word3']],
            'created_at': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'event_id': [0, 1, 0]
        })

# 替换DatasetLoader为MockDatasetLoader
@pytest.fixture
def mock_dataset():
    return MockDatasetLoader().load_data()

# 测试UCLSED类的初始化
def test_uclsed_init(mock_dataset):
    args = args_define().args
    uclsed = UCLSED(args, mock_dataset)
    assert uclsed.save_path is None
    assert uclsed.test_indices is None
    assert uclsed.val_indices is None
    assert uclsed.train_indices is None
    assert uclsed.mask_path is None
    assert uclsed.labels is None
    assert uclsed.times is None
    assert uclsed.g_dict is None
    assert uclsed.views is None
    assert uclsed.features is None

# 测试Preprocessor类的初始化
def test_preprocessor_init(mock_dataset):
    preprocessor = Preprocessor(mock_dataset)
    assert preprocessor is not None

# 测试Preprocessor类的load_data方法
def test_preprocessor_load_data(mock_dataset):
    preprocessor = Preprocessor(mock_dataset)
    event_df = preprocessor.load_data(mock_dataset)
    assert isinstance(event_df, pd.DataFrame)
    assert event_df.shape == (3, 9)

# 测试Preprocessor类的get_nlp方法
def test_preprocessor_get_nlp():
    preprocessor = Preprocessor(None)
    nlp = preprocessor.get_nlp("English")
    assert nlp is not None
    nlp = preprocessor.get_nlp("French")
    assert nlp is not None

# 测试Preprocessor类的construct_graph方法
def test_preprocessor_construct_graph(mock_dataset, tmpdir):
    args = args_define().args
    args.file_path = str(tmpdir) + '/'
    preprocessor = Preprocessor(mock_dataset)
    preprocessor.construct_graph(mock_dataset)
    assert os.path.exists(str(tmpdir) + '/features.npy')
    assert os.path.exists(str(tmpdir) + '/time.npy')
    assert os.path.exists(str(tmpdir) + '/label.npy')

# 测试UCLSED类的fit方法
def test_uclsed_fit(mock_dataset, tmpdir):
    args = args_define().args
    args.file_path = str(tmpdir) + '/'
    args.save_path = str(tmpdir) + '/'
    uclsed = UCLSED(args, mock_dataset)
    uclsed.fit()
    assert os.path.exists(str(tmpdir) + '/train_indices.pt')
    assert os.path.exists(str(tmpdir) + '/val_indices.pt')
    assert os.path.exists(str(tmpdir) + '/test_indices.pt')

# 测试UCLSED类的detection方法
def test_uclsed_detection(mock_dataset, tmpdir):
    args = args_define().args
    args.file_path = str(tmpdir) + '/'
    args.save_path = str(tmpdir) + '/'
    uclsed = UCLSED(args, mock_dataset)
    uclsed.fit()
    ground_truth, predictions = uclsed.detection()
    assert isinstance(ground_truth, np.ndarray)
    assert isinstance(predictions, np.ndarray)

# 测试UCLSED类的evaluate方法
def test_uclsed_evaluate(mock_dataset, tmpdir):
    args = args_define().args
    args.file_path = str(tmpdir) + '/'
    args.save_path = str(tmpdir) + '/'
    uclsed = UCLSED(args, mock_dataset)
    uclsed.fit()
    ground_truth, predictions = uclsed.detection()
    val_f1, val_acc = uclsed.evaluate(ground_truth, predictions)
    assert isinstance(val_f1, float)
    assert isinstance(val_acc, float)