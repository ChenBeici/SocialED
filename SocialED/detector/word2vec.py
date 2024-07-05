import argparse
import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn import metrics
import logging

# Setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class args_define():
    parser = argparse.ArgumentParser()
    
    # Word2Vec parameters
    parser.add_argument('--vector_size', default=100, type=int,
                        help="Dimensionality of the word vectors.")
    parser.add_argument('--window', default=5, type=int,
                        help="Maximum distance between the current and predicted word within a sentence.")
    parser.add_argument('--min_count', default=1, type=int,
                        help="Ignores all words with total frequency lower than this.")
    parser.add_argument('--sg', default=1, type=int,
                        help="Training algorithm: 1 for skip-gram; otherwise CBOW.")
    parser.add_argument('--model_path', default='./word2vec/word2vec_model.model', type=str,
                        help="Path to save/load the Word2Vec model.")
    
    # Other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=False,
                        action='store_true', help="Use cuda")
    parser.add_argument('--data_path', default='./incremental_test_100messagesperday/', type=str,
                        help="Path of features, labels and edges")
    parser.add_argument('--mask_path', default=None, type=str, help="File path that contains the training, validation and test masks")
    parser.add_argument('--resume_path', default=None, type=str,
                        help="File path that contains the partially performed experiment that needs to be resume.")
    parser.add_argument('--resume_point', default=0, type=int,
                        help="The block model to be loaded.")
    parser.add_argument('--resume_current', dest='resume_current', default=True,
                        action='store_false',
                        help="If true, continue to train the resumed model of the current block(to resume a partially trained initial/maintenance block); if false, start the next(infer/predict) block from scratch;")
    parser.add_argument('--log_interval', default=10, type=int,
                        help="Log interval")

    args = parser.parse_args()

class Word2VecModel:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.vector_size = self.args.vector_size
        self.window = self.args.window
        self.min_count = self.args.min_count
        self.sg = self.args.sg
        self.model_path = self.args.model_path
        self.df = None
        self.train_df = None
        self.test_df = None
        self.word2vec_model = None

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
        df['processed_text'] = df['filtered_words'].apply(lambda x: [str(word).lower() for word in x] if isinstance(x, list) else [])
        self.df = df
        return df

    def fit(self):
        """
        Train the Word2Vec model and save it to a file.
        """
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self.train_df = train_df
        self.test_df = test_df

        sentences = train_df['processed_text'].tolist()

        logging.info("Training Word2Vec model...")
        word2vec_model = Word2Vec(sentences=sentences, vector_size=self.vector_size, window=self.window,
                                  min_count=self.min_count, sg=self.sg)
        logging.info("Word2Vec model trained successfully.")
        
        # Save the trained model to a file
        word2vec_model.save(self.model_path)
        logging.info(f"Word2Vec model saved to {self.model_path}")
        
        self.word2vec_model = word2vec_model
        return word2vec_model

    def load_model(self):
        """
        Load the Word2Vec model from a file.
        """
        logging.info(f"Loading Word2Vec model from {self.model_path}...")
        word2vec_model = Word2Vec.load(self.model_path)
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
        test_vectors = np.stack(test_vectors)
        

        return test_vectors

    def evaluate_model(self):
        """
        Evaluate the model.
        """
        test_vectors = self.detection()
        ground_truths = self.test_df['event_id'].tolist()
        
        # Example evaluation: clustering and calculating metrics (this is just a placeholder)
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=len(set(ground_truths)), random_state=42)
        predicted_labels = kmeans.fit_predict(test_vectors)

        # Calculate Adjusted Rand Index (ARI)
        ari = metrics.adjusted_rand_score(ground_truths, predicted_labels)
        print(f"Adjusted Rand Index (ARI): {ari}")

        # Calculate Adjusted Mutual Information (AMI)
        ami = metrics.adjusted_mutual_info_score(ground_truths, predicted_labels)
        print(f"Adjusted Mutual Information (AMI): {ami}")

        # Calculate Normalized Mutual Information (NMI)
        nmi = metrics.normalized_mutual_info_score(ground_truths, predicted_labels)
        print(f"Normalized Mutual Information (NMI): {nmi}")

        return ari, ami, nmi

# Main function
if __name__ == "__main__":
    from Event2012 import Event2012_Dataset

    dataset = Event2012_Dataset.load_data()
    args = args_define.args

    word2vec_model = Word2VecModel(args, dataset)
    
    # Data preprocessing
    word2vec_model.preprocess()
    
    # Train the model
    word2vec_model.fit()
    
    # detection
    predictions = word2vec_model.detection()
    print(predictions)
