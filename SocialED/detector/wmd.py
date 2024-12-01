import os
import pandas as pd
import numpy as np
import torch
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tqdm import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataloader import DatasetLoader


# event_id, filtered_words
class WMD:
    def __init__(self,
                 dataset=DatasetLoader("arabic_twitter").load_data(),
                 vector_size=100,
                 window=5,
                 min_count=1,
                 sg=1,
                 file_path='../model/model_saved/WMD/'):
        self.dataset = dataset
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.file_path = file_path
        self.df = None
        self.train_df = None
        self.test_df = None
        self.word2vec_model = None
        self.model_path = os.path.join(self.file_path, 'word2vec_model.model')

    def preprocess(self):
        """
        Data preprocessing: tokenization, stop words removal, etc.
        """
        df = self.dataset[['filtered_words', 'event_id']].copy()
        df['processed_text'] = df['filtered_words'].apply(
            lambda x: [str(word).lower() for word in x] if isinstance(x, list) else [])
        self.df = df
        return df

    def fit(self):
        """
        Train the Word2Vec model and save it to a file.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        train_df, test_df = train_test_split(self.df, test_size=0.0001, random_state=42)
        self.train_df = train_df
        self.test_df = test_df

        sentences = train_df['processed_text'].tolist()

        print("Training Word2Vec model...")
        word2vec_model = Word2Vec(sentences=sentences, vector_size=self.vector_size, window=self.window,
                                  min_count=self.min_count, sg=self.sg)
        print("Word2Vec model trained successfully.")

        # Save the trained model to a file
        word2vec_model.save(self.model_path)
        print(f"Word2Vec model saved to {self.model_path}")

        self.word2vec_model = word2vec_model.wv  # Use the KeyedVectors part of the Word2Vec model

    def detection(self):
        """
        Detect events using WMD by calculating the distance between messages.
        """
        # Load the saved Word2Vec model
        if self.word2vec_model is None:
            word2vec_model = Word2Vec.load(self.model_path)
            self.word2vec_model = word2vec_model.wv

        test_corpus = self.test_df['processed_text'].tolist()
        train_corpus = self.train_df['processed_text'].tolist()

        print("Calculating WMD distances...")
        instance = WmdSimilarity(train_corpus, self.word2vec_model, num_best=1)

        # Calculate distances and store only the minimum distance for each document
        predictions = []
        for doc in tqdm(test_corpus, desc="Processing documents"):
            distances = instance[doc]
            predictions.append(self.train_df.iloc[distances[0][0]]['event_id'])

        ground_truths = self.test_df['event_id'].tolist()

        return ground_truths, predictions

    def evaluate(self, ground_truths, predictions):
        """
        Evaluate the model.
        """
        # Calculate Normalized Mutual Information (NMI)
        nmi = metrics.normalized_mutual_info_score(ground_truths, predictions)
        print(f"Normalized Mutual Information (NMI): {nmi}")

        # Calculate Adjusted Mutual Information (AMI)
        ami = metrics.adjusted_mutual_info_score(ground_truths, predictions)
        print(f"Adjusted Mutual Information (AMI): {ami}")

        # Calculate Adjusted Rand Index (ARI)
        ari = metrics.adjusted_rand_score(ground_truths, predictions)
        print(f"Adjusted Rand Index (ARI): {ari}")


# Main function
if __name__ == "__main__":
    wmd = WMD()

    # Data preprocessing
    wmd.preprocess()

    # Train the model
    wmd.fit()

    # Detection
    ground_truths, predictions = wmd.detection()

    # Evaluate
    wmd.evaluate(ground_truths, predictions)
