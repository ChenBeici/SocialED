import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn import metrics
import logging
import datetime
import pickle

# Setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class GloVe:
    def __init__(self, dataset, num_clusters=50, random_state=1, file_path='../model_saved/GloVe/', model='../model_needed/glove.6B.100d.txt'):
        self.dataset = dataset
        self.num_clusters = num_clusters
        self.random_state = random_state
        self.model_path = os.path.join(file_path, 'kmeans_model')
        self.df = None
        self.train_df = None
        self.test_df = None
        self.model = model
        self.embeddings_index = self.load_glove_vectors()

    def load_glove_vectors(self):
        """
        Load GloVe pre-trained word vectors.
        """
        embeddings_index = {}
        with open(self.model, 'r', encoding='utf8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        return embeddings_index

    def preprocess(self):
        """
        Data preprocessing: tokenization, stop words removal, etc.
        """
        df = self.dataset
        df['processed_text'] = df['filtered_words'].apply(lambda x: [str(word).lower() for word in x] if isinstance(x, list) else [])
        self.df = df
        return df

    def text_to_glove_vector(self, text, embedding_dim=100):
        """
        Convert text to GloVe vector representation.
        """
        words = text
        embedding = np.zeros(embedding_dim)
        valid_words = 0
        for word in words:
            if word in self.embeddings_index:
                embedding += self.embeddings_index[word]
                valid_words += 1
        if valid_words > 0:
            embedding /= valid_words
        return embedding

    def create_vectors(self, df, text_column):
        """
        Create GloVe vectors for each document.
        """
        texts = df[text_column].tolist()
        vectors = np.array([self.text_to_glove_vector(text) for text in texts])
        return vectors

    def load_model(self):
        """
        Load the KMeans model from a file.
        """
        logging.info(f"Loading KMeans model from {self.model_path}...")
        kmeans_model = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        kmeans_model = kmeans_model.fit(self.train_vectors)  # 重新训练模型
        logging.info("KMeans model loaded successfully.")
        
        self.kmeans_model = kmeans_model
        return kmeans_model

    def fit(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=self.random_state)
        self.train_df = train_df
        self.test_df = test_df
        self.train_vectors = self.create_vectors(train_df, 'processed_text')

        logging.info("Training KMeans model...")
        kmeans_model = KMeans(n_clusters=self.num_clusters, random_state=self.random_state)
        kmeans_model.fit(self.train_vectors)
        logging.info("KMeans model trained successfully.")
        
        # Save the trained model to a file
        with open(self.model_path, 'wb') as f:
            pickle.dump(kmeans_model, f)
        logging.info(f"KMeans model saved to {self.model_path}")

    def detection(self):
        """
        Assign clusters to each document.
        """
        self.load_model()  # Ensure the model is loaded before making detections
        self.test_vectors = self.create_vectors(self.test_df, 'processed_text')
        labels = self.kmeans_model.predict(self.test_vectors)

        # Get the ground truth labels and predicted labels
        ground_truths = self.test_df['event_id'].tolist()
        predicted_labels = labels.tolist()
        return ground_truths, predicted_labels

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
    from data_sets import Event2012_Dataset, Event2018_Dataset, MAVEN_Dataset, Arabic_Dataset

    dataset = Event2012_Dataset.load_data()

    glove = GloVe(dataset)

    # Data preprocessing
    glove.preprocess()
    
    # Train the KMeans model
    glove.fit()
    
    # Detection
    ground_truths, predicted_labels = glove.detection()

    # Evaluate the model
    glove.evaluate(ground_truths, predicted_labels)
