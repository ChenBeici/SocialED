import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn import metrics
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataloader import DatasetLoader
# Setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class SBERT:
    def __init__(self,
                 dataset,
                 model_name='../model/model_needed/paraphrase-MiniLM-L6-v2',
                 df=None,
                 train_df=None,
                 test_df=None, ):
        self.dataset = dataset
        if os.path.exists(model_name):
            self.model_name = model_name
        else:
            self.model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
        self.df = df
        self.train_df = train_df
        self.test_df = test_df
        self.model = SentenceTransformer(self.model_name)

    def preprocess(self):
        """
        Data preprocessing: tokenization, stop words removal, etc.
        """
        df = self.dataset
        df['processed_text'] = df['filtered_words'].apply(
            lambda x: ' '.join([str(word).lower() for word in x]) if isinstance(x, list) else '')
        self.df = df
        return df

    def get_sbert_embeddings(self, text):
        """
        Get SBERT embeddings for a given text.
        """
        return self.model.encode(text)

    def detection(self):
        """
        Detect events by comparing SBERT embeddings.
        """
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self.train_df = train_df
        self.test_df = test_df

        logging.info("Calculating SBERT embeddings for the training set...")
        train_df['sbert_embedding'] = train_df['processed_text'].apply(self.get_sbert_embeddings)
        logging.info("SBERT embeddings calculated for the training set.")

        logging.info("Calculating SBERT embeddings for the test set...")
        test_df['sbert_embedding'] = test_df['processed_text'].apply(self.get_sbert_embeddings)
        logging.info("SBERT embeddings calculated for the test set.")

        train_embeddings = np.stack(self.train_df['sbert_embedding'].values)
        test_embeddings = np.stack(self.test_df['sbert_embedding'].values)

        predictions = []
        for test_emb in test_embeddings:
            distances = np.linalg.norm(train_embeddings - test_emb, axis=1)
            closest_idx = np.argmin(distances)
            predictions.append(self.train_df.iloc[closest_idx]['event_id'])

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
    from dataset.dataloader import Event2012
    dataset = Event2012().load_data()
    sbert = SBERT(dataset)

    sbert.preprocess()
    ground_truths, predictions = sbert.detection()
    sbert.evaluate(ground_truths, predictions)
