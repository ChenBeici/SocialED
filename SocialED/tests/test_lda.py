import pytest
from unittest.mock import MagicMock, patch
from SocialED.detector import LDA
import pandas as pd
import numpy as np

# 示例数据，用于测试
@pytest.fixture
def example_dataset():
    data = {
        'filtered_words': [['event', 'detection', 'test'], ['another', 'test', 'sample']],
        'event_id': [1, 0]
    }
    df = pd.DataFrame(data)
    return df

@pytest.fixture
def lda_instance(example_dataset):
    lda = LDA(dataset=example_dataset)
    return lda

def test_preprocess(lda_instance):
    processed_df = lda_instance.preprocess()
    assert 'processed_text' in processed_df.columns
    assert len(processed_df['processed_text'].iloc[0]) > 0

def test_create_corpus(lda_instance, example_dataset):
    corpus, dictionary = lda_instance.create_corpus(example_dataset, 'processed_text')
    assert isinstance(corpus, list)
    assert len(corpus) == len(example_dataset)
    assert isinstance(dictionary, corpora.Dictionary)

@patch('socialED.detector.lda.LdaModel')
def test_fit(mock_lda_model, lda_instance):
    mock_lda_model.return_value.save = MagicMock()

    lda_instance.preprocess()
    lda_instance.fit()
    
    mock_lda_model.assert_called_once()
    mock_lda_model.return_value.save.assert_called_once_with(lda_instance.model_path)

@patch('socialED.detector.lda.LdaModel')
def test_load_model(mock_lda_model, lda_instance):
    mock_lda_model.load = MagicMock()
    lda_instance.load_model()
    mock_lda_model.load.assert_called_once_with(lda_instance.model_path)

def test_detection(lda_instance):
    lda_instance.load_model = MagicMock()
    lda_instance.create_corpus = MagicMock(return_value=(None, None))
    lda_instance.lda_model = MagicMock()
    lda_instance.lda_model.get_document_topics = MagicMock(return_value=[(0, 0.9), (1, 0.1)])    
    lda_instance.test_df = lda_instance.df.copy()

    ground_truths, predictions = lda_instance.detection()

    assert len(ground_truths) == len(predictions)

@patch('builtins.open', new_callable=MagicMock)
def test_evaluate(mock_open, lda_instance):
    lda_instance.evaluate([1, 0], [1, 1])
    mock_open.assert_called_once_with(lda_instance.model_path + "_evaluation.txt", "a")



