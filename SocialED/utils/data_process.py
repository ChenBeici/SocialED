# -*- coding: utf-8 -*-
"""Data processing utilities for multilingual social media data."""

import numpy as np
import torch
import os
from sklearn.preprocessing import normalize
import pandas as pd
import en_core_web_lg
from datetime import datetime
from torch.utils.data import Dataset
from scipy import sparse
import networkx as nx
from dgl.data.utils import save_graphs
from dgl.data.utils import load_graphs
from collections import Counter
from time import time
import requests
from tqdm import tqdm
from statistics import mean

def construct_graph(df, G=None):
    """Construct a graph from a DataFrame containing social media data.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing social media data with columns:
        tweet_id, user_mentions, user_id, entities, sampled_words
    G : networkx.Graph, optional (default=None)
        Existing graph to add nodes/edges to. If None, creates new graph.
        
    Returns
    -------
    G : networkx.Graph
        Graph with nodes for tweets, users, entities and words, and edges between them.
    """
    import networkx as nx
    
    if G is None:
        G = nx.Graph()
        
    for _, row in df.iterrows():
        # Add tweet node
        tid = 't_' + str(row['tweet_id'])
        G.add_node(tid)
        G.nodes[tid]['tweet_id'] = True

        # Add user nodes
        user_ids = row['user_mentions']
        user_ids.append(row['user_id']) 
        user_ids = ['u_' + str(each) for each in user_ids]
        G.add_nodes_from(user_ids)
        for each in user_ids:
            G.nodes[each]['user_id'] = True

        # Add entity nodes
        entities = row['entities']
        G.add_nodes_from(entities)
        for each in entities:
            G.nodes[each]['entity'] = True

        # Add word nodes
        words = ['w_' + each for each in row['sampled_words']]
        G.add_nodes_from(words)
        for each in words:
            G.nodes[each]['word'] = True

        # Add edges between tweet and other nodes
        edges = []
        edges += [(tid, each) for each in user_ids]
        edges += [(tid, each) for each in entities] 
        edges += [(tid, each) for each in words]
        G.add_edges_from(edges)

    return G

def load_data(name, cache_dir=None):
    """
    Data loading function that downloads .npy files from SocialED_datasets repository.

    Parameters
    ----------
    name : str
        The name of the dataset.
    cache_dir : str, optional
        The directory for dataset caching.
        Default: ``None``.

    Returns
    -------
    data : numpy.ndarray
        The loaded dataset.


    """

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser('~'), '.socialed/data')
    file_path = os.path.join(cache_dir, name + '.npy')

    if os.path.exists(file_path):
        data = np.load(file_path, allow_pickle=True)
    else:
        url = "https://github.com/ChenBeici/SocialED_datasets/raw/main/npy_data/" + name + ".npy"
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        r = requests.get(url, stream=True)
        if r.status_code != 200:
            raise RuntimeError("Failed downloading url %s" % url)
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
        data = np.load(file_path, allow_pickle=True)
    return data


def graph_statistics(G, save_path):
    message = '\nGraph statistics:\n'
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    ave_degree = (num_edges / 2) // num_nodes
    in_degrees = G.in_degrees()
    isolated_nodes = torch.zeros([in_degrees.size()[0]], dtype=torch.long)
    isolated_nodes = (in_degrees == isolated_nodes)
    torch.save(isolated_nodes, save_path + '/isolated_nodes.pt')
    num_isolated_nodes = torch.sum(isolated_nodes).item()

    message += 'We have ' + str(num_nodes) + ' nodes.\n'
    message += 'We have ' + str(num_edges / 2) + ' in-edges.\n'
    message += 'Average degree: ' + str(ave_degree) + '\n'
    message += 'Number of isolated nodes: ' + str(num_isolated_nodes) + '\n'
    print(message)
    with open(save_path + "/graph_statistics.txt", "a") as f:
        f.write(message)

    return num_isolated_nodes


def documents_to_features(df):
    nlp = en_core_web_lg.load()
    features = df.filtered_words.apply(lambda x: nlp(' '.join(x)).vector).values
    return np.stack(features, axis=0)


def extract_time_feature(t_str):
    t = datetime.fromisoformat(str(t_str))
    OLE_TIME_ZERO = datetime(1899, 12, 30)
    delta = t - OLE_TIME_ZERO
    return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]  # 86,400 seconds in day



def get_word2id_emb(wordpath,embpath):
    word2id = {}
    with open(wordpath, 'r') as f:
        for i, w in enumerate(list(f.readlines()[0].split())):
            word2id[w] = i
    embeddings = np.load(embpath)
    return word2id,embeddings

  
def nonlinear_transform_features(wordpath,embpath,df):
    word2id,embeddings = get_word2id_emb(wordpath,embpath)
    features = df.filtered_words.apply(lambda x: [embeddings[word2id[w]] for w in x])
    f_list = []
    for f in features:
        if len(f) != 0:
            f_list.append(np.mean(f, axis=0))
        else:
            f_list.append(np.zeros((300)))
    features = np.stack(f_list, axis=0)
    print(features.shape)
    return features


def getlinear_transform_features(features,src,tgt):
    W = torch.load("../datasets/LinearTranWeight/spacy_{}_{}/best_mapping.pth".format(src,tgt))
    features = np.matmul(features,W)
    return features

def df_to_t_features(df):
    t_features = np.asarray([extract_time_feature(t_str) for t_str in df['created_at']])
    return t_features


def check_class_sizes(ground_truths, predictions):
    count_true_labels = list(Counter(ground_truths).values())  
    ave_true_size = mean(count_true_labels)
    distinct_predictions = list(Counter(predictions).keys()) 
    count_predictions = list(Counter(predictions).values()) 
    large_classes = [distinct_predictions[i] for i, count in enumerate(count_predictions) if count > ave_true_size]
    return large_classes

