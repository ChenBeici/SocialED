import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
from nltk.corpus import stopwords
import nltk
import logging
from scipy.sparse import lil_matrix
import os

# Setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Download stopwords
nltk.download('stopwords')
nltk.download('punkt')

class args_define():
    parser = argparse.ArgumentParser()
    # Hyper-parameters for Word2Vec
    parser.add_argument('--vector_size', default=100, type=int,
                        help="Dimensionality of the word vectors.")
    parser.add_argument('--window', default=5, type=int,
                        help="Maximum distance between the current and predicted word within a sentence.")
    parser.add_argument('--min_count', default=1, type=int,
                        help="Ignores all words with total frequency lower than this.")
    parser.add_argument('--workers', default=4, type=int,
                        help="Number of worker threads to train the model.")
    parser.add_argument('--epochs', default=10, type=int,
                        help="Number of iterations (epochs) over the corpus.")
    parser.add_argument('--num_clusters', default=10, type=int,
                        help="Number of clusters for document clustering.")
    parser.add_argument('--model_path', default='./WMD/word2vec.model', type=str,
                        help="Path to save/load the Word2Vec model.")
    # Other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=False,
                        action='store_true',
                        help="Use cuda")
    parser.add_argument('--data_path', default='./WMD_test/',
                        type=str, help="Path of features, labels and edges")
    parser.add_argument('--mask_path', default=None,
                        type=str, help="File path that contains the training, validation and test masks")
    parser.add_argument('--resume_path', default=None,
                        type=str,
                        help="File path that contains the partially performed experiment that needs to be resume.")
    parser.add_argument('--resume_point', default=0, type=int,
                        help="The block model to be loaded.")
    parser.add_argument('--resume_current', dest='resume_current', default=True,
                        action='store_false',
                        help="If true, continue to train the resumed model of the current block(to resume a partally trained initial/mantenance block);\
                            If false, start the next(infer/predict) block from scratch;")
    parser.add_argument('--log_interval', default=10, type=int,
                        help="Log interval")

    args = parser.parse_args()

class WMDModel:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.vector_size = self.args.vector_size
        self.window = self.args.window
        self.min_count = self.args.min_count
        self.workers = self.args.workers
        self.epochs = self.args.epochs
        self.num_clusters = self.args.num_clusters
        self.model_path = self.args.model_path

    def preprocess(self):
        """
        Data preprocessing: tokenization, stop words removal, etc.
        """
        df = pd.DataFrame(self.dataset, columns=[
            "event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",
            "place_type", "place_full_name", "place_country_code", "hashtags", 
            "user_mentions", "image_urls", "entities", "words", "filtered_words", 
            "sampled_words"
        ])
        stop_words = set(stopwords.words('english'))
        
        def process_text(text):
            # Tokenization, lowercase, remove stopwords
            return [word for word in nltk.word_tokenize(text.lower()) if word.isalpha() and word not in stop_words]
        
        df['processed_text'] = df['filtered_words'].apply(lambda x: process_text(' '.join(x)) if isinstance(x, list) else process_text(x))
        self.df = df
        return df

    def create_corpus(self, df, text_column):
        """
        Create corpus and dictionary required for Word2Vec model.
        """
        texts = df[text_column].tolist()
        return texts

    def fit(self):
        """
        Train the Word2Vec model and save it to a file.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self.train_df = train_df
        self.test_df = test_df
        train_texts = self.create_corpus(train_df, 'processed_text')

        logging.info("Training Word2Vec model...")
        model = Word2Vec(sentences=train_texts, vector_size=self.vector_size, window=self.window, min_count=self.min_count, workers=self.workers, epochs=self.epochs)
        logging.info("Completed training Word2Vec model.")
        
        # Save the trained model to a file
        model.save(self.model_path)
        logging.info(f"Word2Vec model saved to {self.model_path}")
        
        self.word2vec_model = model
        return model

    def load_model(self):
        """
        Load the Word2Vec model from a file.
        """
        logging.info(f"Loading Word2Vec model from {self.model_path}...")
        model = Word2Vec.load(self.model_path)
        logging.info("Word2Vec model loaded successfully.")
        
        self.word2vec_model = model
        return model

    def compute_wmd_distances_sparse(self, model, texts, batch_size=1000):
        """
        Compute WMD distances between documents, batch processing to reduce memory usage, and store results in sparse matrix.
        """
        logging.info("Computing WMD distances...")
        num_texts = len(texts)
        distance_matrix = lil_matrix((num_texts, num_texts))
        instance = WmdSimilarity(texts, model.wv, num_best=num_texts)

        for i in range(0, num_texts, batch_size):
            for j in range(0, num_texts, batch_size):
                end_i = min(i + batch_size, num_texts)
                end_j = min(j + batch_size, num_texts)
                for k in range(i, end_i):
                    if i == j:
                        similarities = instance[texts[k]]
                        for l, distance in similarities:
                            if l < end_j:  # Ensure index l is within current batch
                                distance_matrix[k, l] = distance
                    else:
                        for l in range(j, end_j):
                            if k != l:
                                distance = model.wv.wmdistance(texts[k], texts[l])
                                distance_matrix[k, l] = distance

        logging.info("Completed computing WMD distances.")
        return distance_matrix

    def cluster_documents(self, distance_matrix):
        """
        Cluster documents.
        """
        logging.info("Clustering documents...")
        clustering_model = AgglomerativeClustering(n_clusters=self.num_clusters, affinity='precomputed', linkage='complete')
        labels = clustering_model.fit_predict(distance_matrix.toarray())
        logging.info("Completed clustering documents.")
        return labels

    def evaluate_model(self):
        """
        Evaluate clustering results.
        """
        topics = self.prediction()
        ground_truths = self.test_df['event_id'].tolist()
        predicted_labels = self.cluster_documents(self.compute_wmd_distances_sparse(self.word2vec_model, self.create_corpus(self.test_df, 'processed_text')))

        # Calculate Adjusted Rand Index (ARI)
        ari = adjusted_rand_score(ground_truths, predicted_labels)
        print(f"Adjusted Rand Index (ARI): {ari}")

        # Calculate Adjusted Mutual Information (AMI)
        ami = adjusted_mutual_info_score(ground_truths, predicted_labels)
        print(f"Adjusted Mutual Information (AMI): {ami}")

        # Calculate Normalized Mutual Information (NMI)
        nmi = normalized_mutual_info_score(ground_truths, predicted_labels)
        print(f"Normalized Mutual Information (NMI): {nmi}")

        return ari, ami, nmi

    def prediction(self):
        """
        Assign topics to each document.
        """
        self.load_model()  # Ensure the model is loaded before making predictions
        corpus = self.create_corpus(self.test_df, 'processed_text')
        distance_matrix = self.compute_wmd_distances_sparse(self.word2vec_model, corpus)
        labels = self.cluster_documents(distance_matrix)
        return labels

# Main function
if __name__ == "__main__":
    from Event2012 import Event2012_Dataset

    dataset = Event2012_Dataset.load_data()
    args = args_define.args

    wmd_model = WMDModel(args, dataset)
    
    # Data preprocessing
    wmd_model.preprocess()
    
    # Train the model
    wmd_model.fit()
    
    # Prediction
    predictions = wmd_model.prediction()
    #print(predictions)
    
    # Evaluate model
    wmd_model.evaluate_model()
