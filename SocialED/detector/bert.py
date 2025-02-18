import os
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn import metrics
import torch
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataloader import DatasetLoader
# Setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class BERT:
    r"""The BERT model for social event detection that uses BERT embeddings to 
    detect events in social media data.

    .. note::
        This detector uses BERT embeddings to identify events in social media data.
        The model requires a dataset object with a load_data() method.

    Parameters
    ----------
    dataset : object
        The dataset object containing social media data.
        Must provide load_data() method that returns the raw data.
    model_name : str, optional
        Path to pretrained BERT model or name from HuggingFace.
        If path doesn't exist, defaults to 'bert-base-uncased'.
        Default: ``'../model/model_needed/bert-base-uncased'``.
    max_length : int, optional
        Maximum sequence length for BERT tokenizer.
        Longer sequences will be truncated.
        Default: ``128``.
    df : pandas.DataFrame, optional
        Preprocessed dataframe. If None, will be created during preprocessing.
        Default: ``None``.
    train_df : pandas.DataFrame, optional
        Training data split. If None, will be created during model fitting.
        Default: ``None``.
    test_df : pandas.DataFrame, optional
        Test data split. If None, will be created during model fitting.
        Default: ``None``.
    """
    def __init__(self,
                 dataset,
                 model_name='../model/model_needed/bert-base-uncased',
                 max_length=128,
                 df=None,
                 train_df=None,
                 test_df=None ):
        self.dataset = dataset.load_data()
        if os.path.exists(model_name):
            self.model_name = model_name
        else:
            self.model_name = 'bert-base-uncased'
        self.max_length = max_length
        self.df = df
        self.train_df = train_df
        self.test_df = test_df
        # self.device = torch.device("cpu")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name).to(self.device)

    def preprocess(self):
        """
        Data preprocessing: tokenization, stop words removal, etc.
        """
        df = self.dataset
        df['processed_text'] = df['filtered_words'].apply(
            lambda x: ' '.join([str(word).lower() for word in x]) if isinstance(x, list) else '')
        self.df = df
        return df

    def get_bert_embeddings(self, text):
        """
        Get BERT embeddings for a given text.
        """
        inputs = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, truncation=True,
                                padding='max_length')
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        mean_embedding = torch.mean(last_hidden_states, dim=1).squeeze().cpu().numpy()
        return mean_embedding
        
    def fit(self):
        pass

    def detection(self):
        """
        Detect events by comparing BERT embeddings.
        """
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self.train_df = train_df
        self.test_df = test_df

        logging.info("Calculating BERT embeddings for the training set...")
        train_df['bert_embedding'] = train_df['processed_text'].apply(self.get_bert_embeddings)
        logging.info("BERT embeddings calculated for the training set.")

        logging.info("Calculating BERT embeddings for the test set...")
        test_df['bert_embedding'] = test_df['processed_text'].apply(self.get_bert_embeddings)
        logging.info("BERT embeddings calculated for the test set.")

        train_embeddings = np.stack(self.train_df['bert_embedding'].values)
        test_embeddings = np.stack(self.test_df['bert_embedding'].values)

        predictions = []
        for test_emb in test_embeddings:
            distances = np.linalg.norm(train_embeddings - test_emb, axis=1)
            closest_idx = np.argmin(distances)
            predictions.append(self.train_df.iloc[closest_idx]['event_id'])

        ground_truths = self.test_df['event_id'].tolist()
        return ground_truths, predictions

    def evaluate(self, ground_truths, predictions):
        """
        Evaluate the BERT-based model.
        """

        # Calculate Adjusted Rand Index (ARI)
        ari = metrics.adjusted_rand_score(ground_truths, predictions)
        print(f"Adjusted Rand Index (ARI): {ari}")

        # Calculate Adjusted Mutual Information (AMI)
        ami = metrics.adjusted_mutual_info_score(ground_truths, predictions)
        print(f"Adjusted Mutual Information (AMI): {ami}")

        # Calculate Normalized Mutual Information (NMI)
        nmi = metrics.normalized_mutual_info_score(ground_truths, predictions)
        print(f"Normalized Mutual Information (NMI): {nmi}")

        return ari, ami, nmi



