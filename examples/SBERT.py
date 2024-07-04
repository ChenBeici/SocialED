import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
import logging
from metrics import Metrics

# Set up logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class args_define():
    parser = argparse.ArgumentParser()
    # Hyper-parameters for SBERT
    parser.add_argument('--batch_size', default=32, type=int,
                        help="Batch size for SBERT embeddings extraction.")
    parser.add_argument('--num_clusters', default=10, type=int,
                        help="Number of clusters for document clustering.")

    # Other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=False,
                        action='store_true',
                        help="Use cuda")
    parser.add_argument('--data_path', default='./SBERT_test/', 
                        type=str, help="Path of features, labels and edges")
    parser.add_argument('--mask_path', default=None,
                        type=str, help="File path that contains the training, validation and test masks")
    parser.add_argument('--resume_path', default=None,
                        type=str,
                        help="File path that contains the partially performed experiment that needs to be resumed.")
    parser.add_argument('--resume_point', default=0, type=int,
                        help="The block model to be loaded.")
    parser.add_argument('--resume_current', dest='resume_current', default=True,
                        action='store_false',
                        help="If true, continue to train the resumed model of the current block (to resume a partially trained initial/maintenance block); \
                            If false, start the next (infer/predict) block from scratch;")
    parser.add_argument('--log_interval', default=10, type=int,
                        help="Log interval")

    args = parser.parse_args()

class SBERTModel:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.batch_size = self.args.batch_size
        self.num_clusters = self.args.num_clusters
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def preprocess(self):
        """
        Preprocess data: tokenize text, remove stopwords, etc.
        """
        df = pd.DataFrame(self.dataset, columns=[
            "event_id", "tweet_id", "text", "user_id", "created_at", "user_loc",
            "place_type", "place_full_name", "place_country_code", "hashtags", 
            "user_mentions", "image_urls", "entities", "words", "filtered_words", 
            "sampled_words"
        ])
        df['processed_text'] = df['filtered_words'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        self.df = df
        return df

    def extract_sbert_embeddings(self, texts):
        """
        Extract text embeddings using SBERT model.
        """
        logging.info("Extracting SBERT embeddings...")
        embeddings = self.model.encode(texts, batch_size=self.batch_size, show_progress_bar=True)
        logging.info("Completed extracting SBERT embeddings.")
        return embeddings

    def cluster_documents(self, embeddings):
        """
        Cluster documents based on their embeddings.
        """
        logging.info("Clustering documents...")
        clustering_model = AgglomerativeClustering(n_clusters=self.num_clusters)
        labels = clustering_model.fit_predict(embeddings)
        logging.info("Completed clustering documents.")
        return labels

    def fit(self):
        """
        Train the SBERT model and cluster the documents.
        """
        # Preprocess data
        self.preprocess()

        # Split into train and test sets
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self.train_df = train_df
        self.test_df = test_df

        # Extract SBERT embeddings
        train_texts = train_df['processed_text'].tolist()
        train_embeddings = self.extract_sbert_embeddings(train_texts)

        # Cluster documents
        self.train_labels = self.cluster_documents(train_embeddings)

        return self.train_labels

    def prediction(self):
        """
        Predict the clusters for the test data.
        """
        # Extract SBERT embeddings
        test_texts = self.test_df['processed_text'].tolist()
        test_embeddings = self.extract_sbert_embeddings(test_texts)

        # Cluster documents
        self.test_labels = self.cluster_documents(test_embeddings)

        return self.test_labels

    def evaluate(self):
        """
        Evaluate the clustering results.
        """
        ground_truths = self.test_df['event_id'].tolist()
        ari = Metrics.adjusted_rand_index(ground_truths, self.test_labels)
        ami = Metrics.adjusted_mutual_info(ground_truths, self.test_labels)
        nmi = Metrics.normalized_mutual_info(ground_truths, self.test_labels)
        logging.info(f"Adjusted Rand Index (ARI): {ari}")
        logging.info(f"Adjusted Mutual Information (AMI): {ami}")
        logging.info(f"Normalized Mutual Information (NMI): {nmi}")
        return ari, ami, nmi

# Main function
if __name__ == "__main__":
    from Event2012 import Event2012_Dataset

    dataset = Event2012_Dataset.load_data()
    args = args_define.args

    sbert_model = SBERTModel(args, dataset)
    
    # Train the model
    sbert_model.fit()
    
    # Prediction
    predictions = sbert_model.prediction()
    print(predictions)
    
    # Evaluate model
    sbert_model.evaluate()
