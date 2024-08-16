import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.autograd import Variable
import os
import pandas as pd
from collections import Counter
from itertools import combinations
from time import time
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import en_core_web_lg
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.model_selection import train_test_split
import argparse
import logging

class BiLSTM:
    def __init__(self, dataset, 
                 lr=1e-3, 
                 batch_size=1000, 
                 dropout_keep_prob=0.8, 
                 embedding_size=300, 
                 max_size=5000, 
                 seed=1, 
                 num_hidden_nodes=32, 
                 hidden_dim2=64, 
                 num_layers=1, 
                 bi_directional=True, 
                 pad_index=0, 
                 num_epochs=20, 
                 margin=3, 
                 max_len=10, 
                 file_path='../model_saved/Bilstm/'):
        self.dataset = dataset
        self.lr = lr
        self.batch_size = batch_size
        self.dropout_keep_prob = dropout_keep_prob
        self.embedding_size = embedding_size
        self.max_size = max_size
        self.seed = seed
        self.num_hidden_nodes = num_hidden_nodes
        self.hidden_dim2 = hidden_dim2
        self.num_layers = num_layers
        self.bi_directional = bi_directional
        self.pad_index = pad_index
        self.num_epochs = num_epochs
        self.margin = margin
        self.max_len = max_len
        self.file_path = file_path

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.df = None
        self.train_df = None
        self.test_df = None
        self.word2idx = None
        self.idx2word = None
        self.weight = None
    # Add the rest of the class methods here, like preprocess, fit, detection, etc.

    def preprocess(self):
        """
        Data preprocessing: tokenization, stop words removal, etc.
        """
        self.split()
        df = self.dataset

        # Tokenize tweets
        f_batch_text = df.iloc[:, 5]
        logging.info("Extracted tweets.")

        # Count unique words (converted to lowercases)
        words = Counter()
        for tweet in f_batch_text.values:
            words.update(w.lower() for w in tweet)

        # Convert words from counter to list (sorted by frequencies from high to low)
        words = [key for key, _ in words.most_common()]
        words = ['_PAD', '_UNK'] + words
        logging.info('Extracted unique words.')

        # Construct a mapping of words to indices and vice versa
        self.word2idx = {o: i for i, o in enumerate(words)}
        self.idx2word = {i: o for i, o in enumerate(words)}
        
        # Save
        os.makedirs(self.file_path, exist_ok=True)
        np.save(self.file_path + 'word2idx.npy', self.word2idx)
        np.save(self.file_path + 'idx2word.npy', self.idx2word)
        logging.info('Constructed and saved word2idx and idx2word maps.')

        # Load
        self.word2idx = np.load(self.file_path + 'word2idx.npy', allow_pickle='TRUE').item()
        logging.info('word2idx map loaded.')

        # Convert tokenized tweets to indices
        df["wordsidx"] = df.words.apply(lambda tweet: [self.word2idx.get(w.lower(), self.word2idx['_UNK']) for w in tweet])
        logging.info('Tokenized tweets in the df to word indices.')

        self.df = df
        return df

    def split(self):
        """
        Split the dataset into training, validation, and test sets.
        """
        train_ratio = 0.7
        test_ratio = 0.2
        val_ratio = 0.1

        df = self.dataset

        train_data, temp_data = train_test_split(df, test_size=(1 - train_ratio), random_state=42)
        test_size = test_ratio / (test_ratio + val_ratio)
        test_data, val_data = train_test_split(temp_data, test_size=test_size, random_state=42)

        os.makedirs(self.file_path + '/split_indices/', exist_ok=True)
        np.save(self.file_path + '/split_indices/train_indices_7170.npy', train_data.index.to_numpy())
        np.save(self.file_path + '/split_indices/test_indices_2048.npy', test_data.index.to_numpy())
        np.save(self.file_path + '/split_indices/val_indices_1024.npy', val_data.index.to_numpy())

        os.makedirs(self.file_path + '/split_data/', exist_ok=True)
        train_data.to_numpy().dump(self.file_path + '/split_data/train_data_7170.npy')
        test_data.to_numpy().dump(self.file_path + '/split_data/test_data_2048.npy')
        val_data.to_numpy().dump(self.file_path + '/split_data/val_data_1024.npy')

        self.train_df = train_data
        self.test_df = test_data
        self.val_df = val_data

        logging.info(f"Data split completed: {len(train_data)} train, {len(test_data)} test, {len(val_data)} validation samples.")

    def load_embeddings(self):
        """
        Load pre-trained word embeddings.
        """
        # Initialize weight matrix with zeros
        self.weight = np.zeros((len(self.word2idx), self.embedding_size), dtype=np.float64)

        # Load pre-trained word2vec model
        start = time()
        nlp = en_core_web_lg.load()
        logging.info('Word2vec model took {:.2f} mins to load.'.format((time() - start) / 60))

        # Update word embeddings to weight
        for i in range(len(self.word2idx)):
            w = self.idx2word.get(i)
            token = nlp(w)
            if token.has_vector:
                self.weight[i] = token.vector
        logging.info('Word embeddings extracted. Shape: {}'.format(self.weight.shape))

        # Save and load word embeddings
        np.save(self.file_path + 'word_embeddings.npy', self.weight)
        logging.info('Word embeddings saved.')
        self.weight = np.load(self.file_path + 'word_embeddings.npy')
        logging.info('Word embeddings loaded. Shape: {}'.format(self.weight.shape))
        self.weight = torch.tensor(self.weight, dtype=torch.float)

    def train(self, model, train_iterator, optimizer, loss_func, log_interval=40):
        """
        Train the BiLSTM model.
        """
        n_batches = len(train_iterator)
        epoch_loss = 0
        for i, batch in enumerate(train_iterator):
            optimizer.zero_grad()
            text, text_lengths = batch['text'] 
            predictions = model(text, text_lengths)
            loss, num_triplets = loss_func(predictions, batch['label'])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if i % log_interval == 0:
                print(f'\tBatch: [{i}/{n_batches} ({100. * (i+1) / n_batches:.0f}%)]\tLoss: {epoch_loss / (i+1):.4f}\tNum_triplets: {num_triplets}')
        return epoch_loss / n_batches

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

    def run_train(self, epochs, model, train_iterator, test_iterator, optimizer, loss_func):
        """
        Run the training and evaluation process for the BiLSTM model.
        """
        all_nmi, all_ami, all_ari, all_predictions, all_labels = [], [], [], [], []
        
        for epoch in range(epochs):
            # Train the model
            start = time()
            print(f'Epoch {epoch}. Training.')
            train_loss = self.train(model, train_iterator, optimizer, loss_func)
            print(f'\tTrain Loss: {train_loss:.4f}')
            print(f'\tThis epoch took {(time() - start)/60:.2f} mins to train.')
            
            # Evaluate the model
            start = time()
            print(f'Epoch {epoch}. Evaluating.')
            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(test_iterator):
                    assert i == 0 # cluster all the test tweets at once
                    text, text_lengths = batch['text']
                    predictions = model(text, text_lengths)

                    assert predictions.shape[0] == batch['label'].shape[0]
                    n_classes = len(set(batch['label'].tolist()))
                    kmeans = KMeans(n_clusters=n_classes, n_init=10, random_state=0).fit(predictions)
                    predictions = kmeans.labels_

                    validate_nmi = metrics.normalized_mutual_info_score(batch['label'], predictions)
                    validate_ami = metrics.adjusted_mutual_info_score(batch['label'], predictions)
                    validate_ari = metrics.adjusted_rand_score(batch['label'], predictions)

            all_nmi.append(validate_nmi)
            all_ami.append(validate_ami)
            all_ari.append(validate_ari)
            all_predictions.append(predictions)
            all_labels.append(batch['label'])

            print(f'\tVal. NMI: {validate_nmi:.4f}')
            print(f'\tVal. AMI: {validate_ami:.4f}')
            print(f'\tVal. ARI: {validate_ari:.4f}')
            print(f'\tThis epoch took {(time() - start)/60:.2f} mins to evaluate.')
        
        return all_nmi, all_ami, all_ari, all_predictions, all_labels

    def fit(self):
        """
        Fit the model on the training data and save the best model.
        """
        self.load_embeddings()

        # Split training and test datasets
        train_mask = list(np.load(self.file_path + '/split_indices/train_indices_7170.npy', allow_pickle=True))
        test_mask = list(np.load(self.file_path + '/split_indices/test_indices_2048.npy', allow_pickle=True))
        
        train_data = VectorizeData(self.df.iloc[train_mask, :].copy().reset_index(drop=True), self.max_len)
        test_data = VectorizeData(self.df.iloc[test_mask, :].copy().reset_index(drop=True), self.max_len)
        
        # Construct training and test iterator
        train_iterator = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        test_iterator = DataLoader(test_data, batch_size=len(test_data), shuffle=True)

        # Loss function
        loss_func = OnlineTripletLoss(self.margin, RandomNegativeTripletSelector(self.margin))

        # Model
        lstm_model = LSTM(self.embedding_size, self.weight, self.num_hidden_nodes, self.hidden_dim2,
                        self.num_layers, self.bi_directional, self.dropout_keep_prob, self.pad_index,
                        self.batch_size)

        # Optimizer
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=self.lr)

        # Train and evaluation
        all_nmi, all_ami, all_ari, all_predictions, all_labels = self.run_train(self.num_epochs, lstm_model, train_iterator, test_iterator, optimizer, loss_func)
        
        best_epoch = [i for i, j in enumerate(all_nmi) if j == max(all_nmi)][0]
        print("all_nmi: ", all_nmi)
        print("all_ami: ", all_ami)
        print("all_ari: ", all_ari)
        print("\nTraining completed. Best results at epoch ", best_epoch)
        
        # Save the best model
        self.best_model_path = os.path.join(self.file_path, "best_model.pth")
        torch.save(lstm_model.state_dict(), self.best_model_path)
        print(f"Best model saved at {self.best_model_path}")
        
        self.best_epoch = best_epoch
        self.best_model = lstm_model

    def detection(self):
        """
        Detect events using the best trained model on the test data.
        """
        # Load the best model
        lstm_model = LSTM(self.embedding_size, self.weight, self.num_hidden_nodes, self.hidden_dim2,
                        self.num_layers, self.bi_directional, self.dropout_keep_prob, self.pad_index,
                        self.batch_size)
        lstm_model.load_state_dict(torch.load(self.best_model_path))
        lstm_model.eval()

        # Load test data
        test_mask = list(np.load(self.file_path + '/split_indices/test_indices_2048.npy', allow_pickle=True))
        test_data = VectorizeData(self.df.iloc[test_mask, :].copy().reset_index(drop=True), self.max_len)
        test_iterator = DataLoader(test_data, batch_size=len(test_data), shuffle=False)

        with torch.no_grad():
            for i, batch in enumerate(test_iterator):
                assert i == 0  # Process all test tweets at once
                text, text_lengths = batch['text']
                predictions = lstm_model(text, text_lengths)

                assert predictions.shape[0] == batch['label'].shape[0]
                n_classes = len(set(batch['label'].tolist()))
                kmeans = KMeans(n_clusters=n_classes, n_init=10, random_state=0).fit(predictions)
                predictions = kmeans.labels_

                ground_truths = batch['label']

        return ground_truths, predictions

class LSTM(nn.Module):
    # define all the layers used in model
    def __init__(self, embedding_dim, weight, lstm_units, hidden_dim , lstm_layers,
                 bidirectional, dropout, pad_index, batch_size):
        super().__init__()
        #self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)
        # use pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(weight, padding_idx = pad_index)
        self.lstm = nn.LSTM(embedding_dim,
                            lstm_units,
                            num_layers=lstm_layers,
                            bidirectional=bidirectional,
                            batch_first=True)
        num_directions = 2 if bidirectional else 1
        self.fc1 = nn.Linear(lstm_units * num_directions, hidden_dim)
        #self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm_layers = lstm_layers
        self.num_directions = num_directions
        self.lstm_units = lstm_units


    def init_hidden(self, batch_size):
        h, c = (Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.lstm_units)),
                Variable(torch.zeros(self.lstm_layers * self.num_directions, batch_size, self.lstm_units)))
        return h, c

    def forward(self, text, text_lengths):
        batch_size = text.shape[0]
        h_0, c_0 = self.init_hidden(batch_size)

        embedded = self.embedding(text)
        packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        # output of shape (batch, seq_len, num_directions * hidden_size): tensor containing the 
        # output features (h_t) from the last layer of the LSTM, for each t.
        output, (h_n, c_n) = self.lstm(packed_embedded, (h_0, c_0)) 
        output_unpacked, output_lengths = pad_packed_sequence(output, batch_first=True)
        # get the hidden state of the last time step 
        out = output_unpacked[:, -1, :]
        rel = self.relu(out)
        dense1 = self.fc1(rel)
        #drop = self.dropout(dense1)
        #preds = self.fc2(drop)
        preds = self.dropout(dense1)
        return preds

class VectorizeData(Dataset):
    def __init__(self, df, max_len):
        
        self.df = df
        self.maxlen = max_len
        self.df["lengths"] = self.df.wordsidx.apply(lambda x: self.maxlen if len(x) > self.maxlen else len(x))
        self.df = self.df[self.df["lengths"] > 0].reset_index(drop=True)
        self.df["wordsidxpadded"] = self.df.wordsidx.apply(self.pad_data)
    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        x = self.df.wordsidxpadded[idx]
        lens = self.df.lengths[idx] # truncated tweet length
        y = self.df.event_id[idx]
        sample = {'text':(x,lens), 'label':y}
        return sample
    
    def pad_data(self, tweet):
        padded = np.zeros((self.maxlen,), dtype = np.int64)
        if len(tweet) > self.maxlen:
            padded[:] = tweet[:self.maxlen]
        else:
            padded[:len(tweet)] = tweet
        return padded

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix

class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError

class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)

def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)

def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


if __name__ == "__main__":
    from data_sets import Event2012_Dataset, Event2018_Dataset, MAVEN_Dataset, Arabic_Dataset

    dataset = MAVEN_Dataset.load_data()

    bilstm = BiLSTM(dataset)

    # Data preprocessing
    bilstm.preprocess()

    bilstm.fit()
    # Detection
    ground_truths, predictions = bilstm.detection()

    # Evaluate the model
    bilstm.evaluate(ground_truths, predictions)