import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
import logging
import os

# 设置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class args_define():
    parser = argparse.ArgumentParser()
    # Hyper-parameters for GloVe
    parser.add_argument('--embedding_dim', default=100, type=int,
                        help="Dimensionality of the GloVe word vectors.")
    parser.add_argument('--glove_path', default='glove.6B.100d.txt', type=str,
                        help="Path to the GloVe word vectors file.")
    parser.add_argument('--num_clusters', default=10, type=int,
                        help="Number of clusters for document clustering.")

    # Other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=False,
                        action='store_true',
                        help="Use cuda")
    parser.add_argument('--data_path', default='./incremental_test_100messagesperday/', #default='./incremental_0808/',
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

class GloveModel:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.embedding_dim = self.args.embedding_dim
        self.glove_path = self.args.glove_path
        self.num_clusters = self.args.num_clusters

    def load_glove_embeddings(self, glove_file):
        """
        Load GloVe pretrained word vectors.
        """
        logging.info("Loading GloVe embeddings...")
        embeddings_index = {}
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
        logging.info("Completed loading GloVe embeddings.")
        return embeddings_index

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
        df['processed_text'] = df['filtered_words'].apply(lambda x: [str(word).lower() for word in x] if isinstance(x, list) else [])
        self.df = df
        return df

    def get_document_vectors(self, embeddings_index, texts, embedding_dim):
        """
        Compute document vectors.
        """
        logging.info("Computing document vectors...")
        vectors = []
        for text in texts:
            word_vectors = [embeddings_index[word] for word in text if word in embeddings_index]
            if word_vectors:
                document_vector = np.mean(word_vectors, axis=0)
            else:
                document_vector = np.zeros(embedding_dim)
            vectors.append(document_vector)
        logging.info("Completed computing document vectors.")
        return vectors

    def cluster_documents(self, vectors):
        """
        Cluster documents.
        """
        logging.info("Clustering documents...")
        kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=20, max_iter=300)
        kmeans.fit(vectors)
        logging.info("Completed clustering documents.")
        return kmeans.labels_



    def fit(self):
        """
        Train the GloVe model and cluster the documents.
        """
        # Preprocess data
        self.preprocess()

        # Split into train and test sets
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self.train_df = train_df
        self.test_df = test_df

        # Load GloVe embeddings
        embeddings_index = self.load_glove_embeddings(self.glove_path)

        # Compute document vectors
        train_vectors = self.get_document_vectors(embeddings_index, train_df['processed_text'].tolist(), self.embedding_dim)

        # Cluster documents
        self.train_labels = self.cluster_documents(train_vectors)

        return self.train_labels

    def prediction(self):
        """
        Predict the clusters for the test data.
        """
        # Compute document vectors
        embeddings_index = self.load_glove_embeddings(self.glove_path)
        test_vectors = self.get_document_vectors(embeddings_index, self.test_df['processed_text'].tolist(), self.embedding_dim)

        # Cluster documents
        self.test_labels = self.cluster_documents(test_vectors)

        return self.test_labels

    def evaluate_model(self, ground_truths, predicted_labels):
        """
        Evaluate clustering results.
        """
        logging.info("Evaluating clustering...")
        ari = adjusted_rand_score(ground_truths, predicted_labels)
        ami = adjusted_mutual_info_score(ground_truths, predicted_labels)
        nmi = normalized_mutual_info_score(ground_truths, predicted_labels)
        logging.info(f"Adjusted Rand Index (ARI): {ari}")
        logging.info(f"Adjusted Mutual Information (AMI): {ami}")
        logging.info(f"Normalized Mutual Information (NMI): {nmi}")
        return ari, ami, nmi

    def evaluate(self):
        """
        Evaluate the clustering results.
        """
        ground_truths = self.test_df['event_id'].tolist()
        return self.evaluate_model(ground_truths, self.test_labels)
    

# Main function
if __name__ == "__main__":
    from Event2012 import Event2012_Dataset

    dataset = Event2012_Dataset.load_data()
    args = args_define.args

    glove_model = GloveModel(args, dataset)
    
    # Train the model
    glove_model.fit()
    
    # Prediction
    predictions = glove_model.prediction()
    print(predictions)
    
    # Evaluate model
    glove_model.evaluate()






