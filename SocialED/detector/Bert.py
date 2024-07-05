import argparse
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
import torch
import logging

# Setup logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class args_define():
    parser = argparse.ArgumentParser()
    # Hyper-parameters for BERT
    parser.add_argument('--max_length', default=128, type=int,
                        help="Maximum length of the text sequences.")
    parser.add_argument('--batch_size', default=32, type=int,
                        help="Batch size for BERT embeddings extraction.")
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

class BERTModel:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.max_length = self.args.max_length
        self.batch_size = self.args.batch_size
        self.num_clusters = self.args.num_clusters
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.args.use_cuda else 'cpu')
        self.model.to(self.device)

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
        df['processed_text'] = df['filtered_words'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
        self.df = df
        return df

    def encode_texts(self, texts):
        """
        Use BERT tokenizer to encode texts.
        """
        logging.info("Encoding texts...")
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=self.max_length, return_tensors='pt')
        logging.info("Completed encoding texts.")
        return encodings

    def extract_bert_embeddings(self, encodings):
        """
        Use BERT model to extract text embeddings.
        """
        logging.info("Extracting BERT embeddings...")
        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        embeddings = []

        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(input_ids), self.batch_size):
                batch_input_ids = input_ids[i:i+self.batch_size]
                batch_attention_mask = attention_mask[i:i+self.batch_size]
                outputs = self.model(batch_input_ids, attention_mask=batch_attention_mask)
                last_hidden_state = outputs.last_hidden_state
                # Get the embeddings of [CLS] token
                cls_embeddings = last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(cls_embeddings)
        
        embeddings = np.vstack(embeddings)
        logging.info("Completed extracting BERT embeddings.")
        return embeddings

    def cluster_documents(self, embeddings):
        """
        Cluster documents.
        """
        logging.info("Clustering documents...")
        clustering_model = AgglomerativeClustering(n_clusters=self.num_clusters)
        labels = clustering_model.fit_predict(embeddings)
        logging.info("Completed clustering documents.")
        return labels

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

    def fit(self):
        """
        Train the BERT model and cluster the documents.
        """
        # Preprocess data
        self.preprocess()

        # Split data into train and test sets
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=42)
        self.train_df = train_df
        self.test_df = test_df

        # Encode texts
        train_texts = train_df['processed_text'].tolist()
        train_encodings = self.encode_texts(train_texts)

        # Extract BERT embeddings
        train_embeddings = self.extract_bert_embeddings(train_encodings)

        # Cluster documents
        self.train_labels = self.cluster_documents(train_embeddings)

        return self.train_labels

    def prediction(self):
        """
        Predict the clusters for the test data.
        """
        # Encode texts
        test_texts = self.test_df['processed_text'].tolist()
        test_encodings = self.encode_texts(test_texts)

        # Extract BERT embeddings
        test_embeddings = self.extract_bert_embeddings(test_encodings)

        # Cluster documents
        self.test_labels = self.cluster_documents(test_embeddings)

        return self.test_labels

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

    bert_model = BERTModel(args, dataset)
    
    # Train the model
    bert_model.fit()
    
    # Prediction
    predictions = bert_model.prediction()
    print(predictions)
    
    # Evaluate model
    bert_model.evaluate()
