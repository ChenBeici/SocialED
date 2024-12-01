import os
import pytest
import pandas as pd
from unittest.mock import patch

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detector.LDA import LDA
from dataset.dataloader import DatasetLoader

from gensim.models.ldamodel import LdaModel
from gensim import corpora

# 设置测试数据路径
TEST_DATA_PATH = os.path.join(os.path.dirname(__file__), 'test_data')


# 创建一个测试数据集
@pytest.fixture
def test_dataset():
    data = {
        'filtered_words': [['word1', 'word2'], ['word3', 'word4'], ['word5', 'word6']],
        'event_id': [1, 2, 3]
    }
    return pd.DataFrame(data)


# 测试预处理方法
def test_preprocess(test_dataset):
    lda = LDA(dataset=test_dataset)
    df = lda.preprocess()
    assert 'processed_text' in df.columns
    assert df['processed_text'].tolist() == [['word1', 'word2'], ['word3', 'word4'], ['word5', 'word6']]


# 测试创建语料库方法
def test_create_corpus(test_dataset):
    lda = LDA(dataset=test_dataset)
    lda.preprocess()
    corpus, dictionary = lda.create_corpus(lda.df, 'processed_text')
    assert isinstance(corpus, list)
    assert isinstance(dictionary, corpora.Dictionary)


# 测试模型训练方法
def test_fit(test_dataset, tmpdir):
    lda = LDA(dataset=test_dataset, file_path=str(tmpdir))
    lda.preprocess()
    lda.fit()
    assert os.path.exists(os.path.join(tmpdir, 'lda_model'))


# 测试模型加载方法
def test_load_model(test_dataset, tmpdir):
    lda = LDA(dataset=test_dataset, file_path=str(tmpdir))
    lda.preprocess()
    lda.fit()
    loaded_model = lda.load_model()
    assert isinstance(loaded_model, LdaModel)


# 测试主题显示方法
def test_display_topics(test_dataset, tmpdir, capsys):
    lda = LDA(dataset=test_dataset, file_path=str(tmpdir))
    lda.preprocess()
    lda.fit()
    lda.load_model()
    lda.display_topics()
    captured = capsys.readouterr()
    assert "Topic" in captured.out


# 测试检测方法
def test_detection(test_dataset, tmpdir):
    lda = LDA(dataset=test_dataset, file_path=str(tmpdir))
    lda.preprocess()
    lda.fit()
    ground_truths, predictions = lda.detection()
    assert isinstance(ground_truths, list)
    assert isinstance(predictions, list)
    assert len(ground_truths) == len(predictions)


# 测试评估方法
def test_evaluate(test_dataset, tmpdir):
    lda = LDA(dataset=test_dataset, file_path=str(tmpdir))
    lda.preprocess()
    lda.fit()
    ground_truths, predictions = lda.detection()
    lda.evaluate(ground_truths, predictions)
    assert os.path.exists(os.path.join(tmpdir, 'lda_model_evaluation.txt'))


# 测试主函数
@patch('dataset.dataloader.DatasetLoader.load_data')
def test_main(mock_load_data, tmpdir):
    # 模拟数据集加载
    mock_load_data.return_value = pd.DataFrame({
        'filtered_words': [['word1', 'word2'], ['word3', 'word4'], ['word5', 'word6']],
        'event_id': [1, 2, 3]
    })

    lda = LDA(dataset=DatasetLoader("event2012").load_data(), file_path=str(tmpdir))
    lda.preprocess()
    lda.fit()
    ground_truths, predictions = lda.detection()
    lda.evaluate(ground_truths, predictions)
    assert os.path.exists(os.path.join(tmpdir, 'lda_model'))
    assert os.path.exists(os.path.join(tmpdir, 'unique_ground_truths_predictions.csv'))
    assert os.path.exists(os.path.join(tmpdir, 'lda_model_evaluation.txt'))
