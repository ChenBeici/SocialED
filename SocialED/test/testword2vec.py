import pytest
import sys
import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Assuming the WORD2VEC class is in a file named word2vec.py
from detector.word2vec import *

# Mock data for testing
@pytest.fixture
def mock_dataset():
    data = {
        'filtered_words': [['word1', 'word2'], ['word3', 'word4'], ['word5', 'word6']],
        'event_id': [1, 2, 1]
    }
    return pd.DataFrame(data)

@pytest.fixture
def word2vec_instance(mock_dataset):
    word2vec = WORD2VEC(dataset=mock_dataset)
    word2vec.preprocess()  # Ensure preprocessing is done before any other method
    return word2vec

def test_preprocess(word2vec_instance):
    df = word2vec_instance.df
    assert isinstance(df, pd.DataFrame)
    assert 'processed_text' in df.columns
    assert df['processed_text'].tolist() == [['word1', 'word2'], ['word3', 'word4'], ['word5', 'word6']]

def test_fit(word2vec_instance, tmpdir):
    word2vec_instance.file_path = os.path.join(tmpdir, 'word2vec_model.model')
    model = word2vec_instance.fit()
    assert isinstance(model, Word2Vec)
    assert os.path.exists(word2vec_instance.file_path)

def test_load_model(word2vec_instance, tmpdir):
    word2vec_instance.file_path = os.path.join(tmpdir, 'word2vec_model.model')
    word2vec_instance.fit()  # Ensure the model is saved
    loaded_model = word2vec_instance.load_model()
    assert isinstance(loaded_model, Word2Vec)

def test_document_vector(word2vec_instance):
    word2vec_instance.fit()  # Ensure the model is trained
    document = ['word1', 'word2']
    vector = word2vec_instance.document_vector(document)
    assert isinstance(vector, np.ndarray)
    assert vector.shape == (word2vec_instance.vector_size,)

def test_detection(word2vec_instance):
    word2vec_instance.fit()  # Ensure the model is trained
    ground_truths, predictions = word2vec_instance.detection()
    assert isinstance(ground_truths, list)
    assert isinstance(predictions, np.ndarray)
    assert len(ground_truths) == len(predictions)

def test_evaluate(word2vec_instance):
    ground_truths = [1, 2, 1]
    predictions = np.array([0, 1, 0])
    ari, ami, nmi = word2vec_instance.evaluate(ground_truths, predictions)
    assert isinstance(ari, float)
    assert isinstance(ami, float)
    assert isinstance(nmi, float)