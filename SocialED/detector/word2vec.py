import argparse
import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn import metrics
import logging
from sklearn.cluster import KMeans
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataloader import DatasetLoader
# Setup logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class WORD2VEC:
    def __init__(self,
                 dataset=DatasetLoader("arabic_twitter").load_data(),
                 vector_size=100, 
                 window=5,
                 min_count=1,
                 sg=1,
                 file_path='../model/model_saved/Word2vec/word2vec_model.model'):
        # print("ASDASd")
        # exit()
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
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self.train_df = train_df
        self.test_df = test_df

        sentences = train_df['processed_text'].tolist()

        logging.info("Training Word2Vec model...")
        word2vec_model = Word2Vec(sentences=sentences, vector_size=self.vector_size, window=self.window,
                                  min_count=self.min_count, sg=self.sg)
        logging.info("Word2Vec model trained successfully.")

        # Save the trained model to a file
        word2vec_model.save(self.file_path)
        logging.info(f"Word2Vec model saved to {self.file_path}")

        self.word2vec_model = word2vec_model
        return word2vec_model

    def load_model(self):
        """
        Load the Word2Vec model from a file.
        """
        logging.info(f"Loading Word2Vec model from {self.file_path}...")
        word2vec_model = Word2Vec.load(self.file_path)
        logging.info("Word2Vec model loaded successfully.")

        self.word2vec_model = word2vec_model
        return word2vec_model

    def document_vector(self, document):
        """
        Create a document vector by averaging the Word2Vec embeddings of its words.
        """
        words = [word for word in document if word in self.word2vec_model.wv]
        if words:
            return np.mean(self.word2vec_model.wv[words], axis=0)
        else:
            return np.zeros(self.vector_size)

    def detection(self):
        """
        Detect events by representing each document as the average Word2Vec embedding of its words.
        """
        self.load_model()  # Ensure the model is loaded before making detections

        test_vectors = self.test_df['processed_text'].apply(self.document_vector)
        predictions = np.stack(test_vectors)

        ground_truths = self.test_df['event_id'].tolist()
        kmeans = KMeans(n_clusters=len(set(ground_truths)), random_state=42)
        predictions = kmeans.fit_predict(predictions)

        return ground_truths, predictions

    def evaluate(self, ground_truths, predictions):
        """
        Evaluate the model.
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


# Main function
if __name__ == "__main__":
    # from data_sets import Event2012_Dataset, Event2018_Dataset, MAVEN_Dataset, Arabic_Dataset


    word2vec = WORD2VEC()

    # Data preprocessing
    word2vec.preprocess()

    # Train the model
    word2vec.fit()

    # detection
    ground_truths, predictions = word2vec.detection()

    # Evaluate the model
    word2vec.evaluate(ground_truths, predictions)
