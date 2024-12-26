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
from detector.bert import *


# Test data preprocessing
def test_preprocess(sample_dataset):
    """Test preprocessing of text data.
    
    Parameters
    ----------
    sample_dataset : DatasetLoader
        Sample dataset for testing preprocessing.
        
    Returns
    -------
    None
        Asserts that processed text column exists and contains strings.
    """
    bert = BERT(sample_dataset)
    df = bert.preprocess()
    assert 'processed_text' in df.columns
    assert df['processed_text'].apply(lambda x: isinstance(x, str)).all()

# Test getting BERT embeddings
def test_get_bert_embeddings(sample_dataset):
    """Test extraction of BERT embeddings from text.
    
    Parameters
    ----------
    sample_dataset : DatasetLoader
        Sample dataset for testing embeddings.
        
    Returns
    -------
    None
        Asserts that embeddings are numpy arrays of correct shape.
    """
    bert = BERT(sample_dataset)
    bert.preprocess()
    text = 'hello world'
    embedding = bert.get_bert_embeddings(text)
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape == (768,)

# Test event detection
def test_detection(sample_dataset):
    """Test event detection functionality.
    
    Parameters
    ----------
    sample_dataset : DatasetLoader
        Sample dataset for testing detection.
        
    Returns
    -------
    None
        Asserts that predictions match ground truth length.
    """
    bert = BERT(sample_dataset)
    bert.preprocess()
    ground_truths, predictions = bert.detection()
    assert len(ground_truths) == len(predictions)

# Test evaluation metrics
def test_evaluate(sample_dataset):
    """Test evaluation of detection results.
    
    Parameters
    ----------
    sample_dataset : DatasetLoader
        Sample dataset for testing evaluation.
        
    Returns
    -------
    None
        Asserts that evaluation metrics are valid floats.
    """
    bert = BERT(sample_dataset)
    bert.preprocess()
    ground_truths, predictions = bert.detection()
    ari, ami, nmi = bert.evaluate(ground_truths, predictions)
    assert isinstance(ari, float)
    assert isinstance(ami, float)
    assert isinstance(nmi, float)

if __name__ == "__main__":
    pytest.main()