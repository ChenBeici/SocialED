# eventx original paper:
# Bang Liu, Fred X Han, Di Niu, Linglong Kong, Kunfeng Lai, and Yu Xu. 2020. Story Forest: Extracting Events and Telling Stories from Breaking News. TKDD 14, 3 (2020), 1–28.
import pandas as pd
import numpy as np
import itertools
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import random
from sklearn import metrics
from collections import Counter
import os
import pickle
from sklearn.model_selection import train_test_split
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class EventX:
    r"""The EventX model for social event detection that extracts events from breaking news
    using keyword co-occurrence and graph-based clustering.

    .. note::
        This detector uses keyword co-occurrence and graph-based clustering to identify events in social media data.
        The model requires a dataset object with a load_data() method.

    See :cite:`liu2020story` for details.

    Parameters
    ----------
    dataset : object
        The dataset object containing social media data.
        Must provide load_data() method that returns the raw data.
    file_path : str, optional
        Path to save model files. Default: ``'../model/model_saved/eventX/'``.
    num_repeats : int, optional
        Number of times to repeat keyword extraction. Default: ``5``.
    min_cooccur_time : int, optional
        Minimum number of times keywords must co-occur. Default: ``2``.
    min_prob : float, optional
        Minimum probability threshold for keyword selection. Default: ``0.15``.
    max_kw_num : int, optional
        Maximum number of keywords to extract per document. Default: ``3``.
    """
    def __init__(self,
                 dataset,
                 file_path='../model/model_saved/eventX/',
                 num_repeats=5,
                 min_cooccur_time=2,
                 min_prob=0.15,
                 max_kw_num=3):
        self.dataset = dataset.load_data()
        self.file_path = file_path
        self.mask_path = None
        self.num_repeats = num_repeats
        self.min_cooccur_time = min_cooccur_time
        self.min_prob = min_prob
        self.max_kw_num = max_kw_num

    # construct offline df
    def preprocess(self):

        self.split()
        self.mask_path = '../model/model_saved/eventX/split_indices/test_indices.npy'
        df = self.dataset
        logging.info("Loaded all_df_words_ents_mid.")

        test_mask = np.load(self.mask_path, allow_pickle=True)
        df = df.iloc[test_mask, :]
        logging.info("Test df extracted.")

        np.save(self.file_path + 'corpus_offline.npy', df.values)
        logging.info("corpus_offline saved.")

        self.df = self.dataset[['event_id','filtered_words','tweet_id','entities']].copy()
        logging.info("Data preprocessed.")

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
        np.save(self.file_path + '/split_indices/train_indices.npy', train_data.index.to_numpy())
        np.save(self.file_path + '/split_indices/test_indices.npy', test_data.index.to_numpy())
        np.save(self.file_path + '/split_indices/val_indices.npy', val_data.index.to_numpy())

        os.makedirs(self.file_path + '/split_data/', exist_ok=True)
        train_data.to_numpy().dump(self.file_path + '/split_data/train_data.npy')
        test_data.to_numpy().dump(self.file_path + '/split_data/test_data.npy')
        val_data.to_numpy().dump(self.file_path + '/split_data/val_data.npy')

        self.train_df = train_data
        self.test_df = test_data
        self.val_df = val_data

        logging.info(
            f"Data split completed: {len(train_data)} train, {len(test_data)} test, {len(val_data)} validation samples.")

    def fit(self):
        pass


    def detection(self):
        kw_pair_dict, kw_dict = self.construct_dict(self.df, self.file_path)
        m_kw_pair_dict, m_kw_dict, map_index_to_kw, map_kw_to_index = self.map_dicts(kw_pair_dict, kw_dict,
                                                                                     self.file_path)
        G = self.construct_kw_graph(kw_pair_dict, kw_dict, self.min_cooccur_time, self.min_prob)
        communities = []
        self.detect_kw_communities_iter(G, communities, kw_pair_dict, kw_dict, self.max_kw_num)
        m_communities = self.map_communities(communities, map_kw_to_index)

        logging.info("Model fitted.")
        m_tweets, ground_truths = self.map_tweets(self.df, self.file_path)
        predictions = self.classify_docs(m_tweets, m_communities, map_kw_to_index, self.file_path)

        logging.info("Events detected.")
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

    def construct_dict(self, df, dir_path=None):
        kw_pair_dict = {}
        kw_dict = {}

        for _, row in df.iterrows():
            tweet_id = str(row['tweet_id'])
            entities = row['entities']
            entities = ['_'.join(tup) for tup in entities]
            for each in entities:
                if each not in kw_dict.keys():
                    kw_dict[each] = []
                kw_dict[each].append(tweet_id)

            words = row['filtered_words']
            for each in words:
                if each not in kw_dict.keys():
                    kw_dict[each] = []
                kw_dict[each].append(tweet_id)

            for r in itertools.combinations(entities + words, 2):
                r = list(r)
                r.sort()
                pair = (r[0], r[1])
                if pair not in kw_pair_dict.keys():
                    kw_pair_dict[pair] = []
                kw_pair_dict[pair].append(tweet_id)

        if dir_path is not None:
            pickle.dump(kw_dict, open(dir_path + '/kw_dict.pickle', 'wb'))
            pickle.dump(kw_pair_dict, open(dir_path + '/kw_pair_dict.pickle', 'wb'))

        return kw_pair_dict, kw_dict

    def map_dicts(self, kw_pair_dict, kw_dict, dir_path=None):
        map_index_to_kw = {}
        m_kw_dict = {}
        for i, k in enumerate(kw_dict.keys()):
            map_index_to_kw['k' + str(i)] = k
            m_kw_dict['k' + str(i)] = kw_dict[k]
        map_kw_to_index = {v: k for k, v in map_index_to_kw.items()}
        m_kw_pair_dict = {}
        for _, pair in enumerate(kw_pair_dict.keys()):
            m_kw_pair_dict[(map_kw_to_index[pair[0]], map_kw_to_index[pair[1]])] = kw_pair_dict[pair]

        if dir_path is not None:
            pickle.dump(m_kw_pair_dict, open(dir_path + '/m_kw_pair_dict.pickle', 'wb'))
            pickle.dump(m_kw_dict, open(dir_path + '/m_kw_dict.pickle', 'wb'))
            pickle.dump(map_index_to_kw, open(dir_path + '/map_index_to_kw.pickle', 'wb'))
            pickle.dump(map_kw_to_index, open(dir_path + '/map_kw_to_index.pickle', 'wb'))

        return m_kw_pair_dict, m_kw_dict, map_index_to_kw, map_kw_to_index

    def construct_kw_graph(self, kw_pair_dict, kw_dict, min_cooccur_time, min_prob):
        G = nx.Graph()
        G.add_nodes_from(list(kw_dict.keys()))
        for pair, co_tid_list in kw_pair_dict.items():
            if len(co_tid_list) > min_cooccur_time:
                if (len(co_tid_list) / len(kw_dict[pair[0]]) > min_prob) and (
                        len(co_tid_list) / len(kw_dict[pair[1]]) > min_prob):
                    G.add_edge(*pair)
        return G

    def detect_kw_communities_iter(self, G, communities, kw_pair_dict, kw_dict, max_kw_num=3):
        connected_components = [c for c in nx.connected_components(G)]
        while len(connected_components) >= 1:
            c = connected_components[0]
            if len(c) < max_kw_num:
                communities.append(c)
                G.remove_nodes_from(c)
            else:
                c_sub_G = G.subgraph(c).copy()
                d = nx.edge_betweenness_centrality(c_sub_G)
                max_value = max(d.values())
                edges = [key for key, value in d.items() if value == max_value]
                if len(edges) > 1:
                    probs = []
                    for e in edges:
                        e = list(e)
                        e.sort()
                        pair = (e[0], e[1])
                        co_len = len(kw_pair_dict[pair])
                        e_prob = (co_len / len(kw_dict[pair[0]]) + co_len / len(kw_dict[pair[1]])) / 2
                        probs.append(e_prob)
                    min_prob = min(probs)
                    min_index = [i for i, j in enumerate(probs) if j == min_prob]
                    edge_to_remove = edges[min_index[0]]
                else:
                    edge_to_remove = edges[0]
                G.remove_edge(*edge_to_remove)
            connected_components = [c for c in nx.connected_components(G)]

    def map_communities(self, communities, map_kw_to_index):
        m_communities = []
        for cluster in communities:
            m_cluster = ' '.join(map_kw_to_index[kw] for kw in cluster)
            m_communities.append(m_cluster)
        return m_communities

    def classify_docs(self, test_tweets, m_communities, map_kw_to_index, dir_path=None):
        m_test_tweets = []
        for doc in test_tweets:
            m_doc = ' '.join(map_kw_to_index[kw] for kw in doc)     
            m_test_tweets.append(m_doc)

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(m_communities + m_test_tweets)
        train_size = len(m_communities)
        test_size = len(m_test_tweets)
        classes = []
        for i in range(test_size):
            cosine_similarities = linear_kernel(X[train_size + i], X[:train_size]).flatten()
            max_similarity = cosine_similarities[cosine_similarities.argsort()[-1]]
            related_clusters = [i for i, sim in enumerate(cosine_similarities) if sim == max_similarity]
            if len(related_clusters) == 1:
                classes.append(related_clusters[0])
            else:
                classes.append(random.choice(related_clusters))

        if dir_path is not None:
            np.save(dir_path + '/classes.npy', classes)

        return classes

    def map_tweets(self, df, dir_path=None):
        m_tweets = []
        ground_truths = []
        for _, row in df.iterrows():
            entities = row['entities']
            entities = ['_'.join(tup) for tup in entities]
            words = row['filtered_words']
            m_tweets.append(entities + words)
            ground_truths.append(row['event_id'])

        if dir_path is not None:
            with open(os.path.join(dir_path, 'm_tweets.pkl'), 'wb') as f:
                pickle.dump(m_tweets, f)
            with open(os.path.join(dir_path, 'ground_truths.pkl'), 'wb') as f:
                pickle.dump(ground_truths, f)

        return m_tweets, ground_truths


# recursive version, can cause RecursionError when running on large graphs. Changed to iterative version below.
def detect_kw_communities(G, communities, kw_pair_dict, kw_dict, max_kw_num=3):
    connected_components = [c for c in nx.connected_components(G)]
    if len(connected_components) >= 1:
        c = connected_components[0]
        if len(c) < max_kw_num:
            communities.append(c)
            G.remove_nodes_from(c)
        else:
            c_sub_G = G.subgraph(c).copy()
            d = nx.edge_betweenness_centrality(c_sub_G)
            max_value = max(d.values())
            edges = [key for key, value in d.items() if value == max_value]
            # If two edges have the same betweenness score, the one with lower conditional probability will be removed
            if len(edges) > 1:
                probs = []
                for e in edges:
                    e = list(e)
                    e.sort()
                    pair = (e[0], e[1])
                    co_len = len(kw_pair_dict[pair])
                    e_prob = (co_len / len(kw_dict[pair[0]]) + co_len / len(kw_dict[pair[1]])) / 2
                    probs.append(e_prob)
                min_prob = min(probs)
                min_index = [i for i, j in enumerate(probs) if j == min_prob]
                edge_to_remove = edges[min_index[0]]
            else:
                edge_to_remove = edges[0]
            G.remove_edge(*edge_to_remove)
        detect_kw_communities(G, communities, kw_pair_dict, kw_dict, max_kw_num)
    else:
        return




