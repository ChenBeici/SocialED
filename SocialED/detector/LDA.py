import argparse
import os
import logging
import datetime
import pandas as pd
import numpy as np
from gensim.models.ldamodel import LdaModel
from gensim import corpora
from sklearn.model_selection import train_test_split
from sklearn import metrics
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataloader import DatasetLoader

class LDA:
    def __init__(self,
                 dataset=DatasetLoader("arabic_twitter").load_data(),
                 num_topics=50,
                 passes=20,
                 iterations=50,
                 alpha='symmetric',
                 eta=None,
                 random_state=1,
                 eval_every=10,
                 chunksize=2000,
                 file_path='../model/model_saved/LDA/'):
        self.dataset = dataset
        self.num_topics = num_topics
        self.passes = passes
        self.iterations = iterations
        self.alpha = alpha
        self.eta = eta
        self.random_state = random_state
        self.eval_every = eval_every
        self.chunksize = chunksize
        self.df = None
        self.train_df = None
        self.test_df = None
        self.file_path = file_path
        self.model_path = os.path.join(file_path, 'lda_model')

    def preprocess(self):
        """
        Data preprocessing: tokenization, stop words removal, etc.
        """
        df = self.dataset[['filtered_words', 'event_id']].copy()
        df['processed_text'] = df['filtered_words'].apply(
            lambda x: [str(word).lower() for word in x] if isinstance(x, list) else [])
        self.df = df
        return df

    def create_corpus(self, df, text_column):
        """
        Create corpus and dictionary required for LDA model.
        """
        texts = df[text_column].tolist()
        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        return corpus, dictionary

    def load_model(self):
        """
        Load the LDA model from a file.
        """
        logging.info(f"Loading LDA model from {self.model_path}...")
        lda_model = LdaModel.load(self.model_path)
        logging.info("LDA model loaded successfully.")

        self.lda_model = lda_model
        return lda_model

    def display_topics(self, num_words=10):
        """
        Display topics generated by the LDA model.
        """
        topics = self.lda_model.show_topics(num_words=num_words, formatted=False)
        for i, topic in topics:
            print(f"Topic {i}: {[word for word, _ in topic]}")

    def fit(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=self.random_state)
        self.train_df = train_df
        self.test_df = test_df
        train_corpus, train_dictionary = self.create_corpus(train_df, 'processed_text')

        logging.info("Training LDA model...")
        lda_model = LdaModel(corpus=train_corpus, id2word=train_dictionary, num_topics=self.num_topics,
                             passes=self.passes,
                             iterations=self.iterations, alpha=self.alpha, eta=self.eta, random_state=self.random_state,
                             eval_every=self.eval_every, chunksize=self.chunksize)
        logging.info("LDA model trained successfully.")

        # Save the trained model to a file
        lda_model.save(self.model_path)
        logging.info(f"LDA model saved to {self.model_path}")

    def detection(self):
        """
        Assign topics to each document and save unique ground truths and predictions to a CSV file.
        """
        self.load_model()  # Ensure the model is loaded before making detections
        corpus, _ = self.create_corpus(self.test_df, 'processed_text')
        topics = [self.lda_model.get_document_topics(bow) for bow in corpus]

        # Get the ground truth labels and predicted labels
        ground_truths = self.test_df['event_id'].tolist()
        predictions = [max(topic, key=lambda x: x[1])[0] for topic in topics]

        # Convert to sets to remove duplicates
        unique_ground_truths = list(set(ground_truths))
        unique_predictions = list(set(predictions))

        # Pad the shorter list with None to make them the same length
        max_len = max(len(unique_ground_truths), len(unique_predictions))
        unique_ground_truths.extend([None] * (max_len - len(unique_ground_truths)))
        unique_predictions.extend([None] * (max_len - len(unique_predictions)))

        # Combine into a dataframe
        data = {
            'Unique Ground Truths': unique_ground_truths,
            'Unique Predictions': unique_predictions
        }
        df = pd.DataFrame(data)

        # Save to a CSV file
        output_file = os.path.join(self.file_path, "unique_ground_truths_predictions.csv")
        df.to_csv(output_file, index=False)
        print(f"Unique ground truths and predictions have been saved to {output_file}")

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

        # Get the current date and time
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Save results to a file in append mode
        with open(self.model_path + "_evaluation.txt", "a") as f:
            f.write(f"Date and Time: {current_datetime}\n")
            f.write(f"Normalized Mutual Information (NMI): {nmi}\n")
            f.write(f"Adjusted Mutual Information (AMI): {ami}\n")
            f.write(f"Adjusted Rand Index (ARI): {ari}\n")
            f.write("\n")  # Add a newline for better readability


# Main function
if __name__ == "__main__":
    dataset = DatasetLoader("maven").load_data()

    lda = LDA(dataset)

    # Data preprocessing
    lda.preprocess()

    # Train the LDA model
    lda.fit()

    # detection
    ground_truths, predictions = lda.detection()

    # Evaluate the model
    lda.evaluate(ground_truths, predictions)