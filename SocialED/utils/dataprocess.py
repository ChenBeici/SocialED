# -*- coding: utf-8 -*-
"""Data processing utilities for multilingual social media data."""

import numpy as np
import torch
import os
import en_core_web_lg
from datetime import datetime
from collections import Counter
import requests
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
    """
    Calculate and save basic statistics of a graph.

    Parameters
    ----------
    G : networkx.Graph
        The input graph to analyze.
    save_path : str
        Directory path to save the statistics.

    Returns
    -------
    num_isolated_nodes : int
        Number of isolated nodes in the graph.
    """
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
    """
    Convert document text to feature vectors using spaCy.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing documents with 'filtered_words' column.

    Returns
    -------
    numpy.ndarray
        Document feature vectors stacked into a matrix.
    """
    nlp = en_core_web_lg.load()
    features = df.filtered_words.apply(lambda x: nlp(' '.join(x)).vector).values
    return np.stack(features, axis=0)

def extract_time_feature(t_str):
    """
    Extract time features from timestamp string.

    Parameters
    ----------
    t_str : str
        Timestamp string in ISO format.

    Returns
    -------
    list
        List containing two normalized time features: [days, seconds].
    """
    t = datetime.fromisoformat(str(t_str))
    OLE_TIME_ZERO = datetime(1899, 12, 30)
    delta = t - OLE_TIME_ZERO
    return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]  # 86,400 seconds in day

def get_word2id_emb(wordpath,embpath):
    """
    Load word-to-id mapping and embeddings from files.

    Parameters
    ----------
    wordpath : str
        Path to file containing words.
    embpath : str
        Path to file containing embeddings.

    Returns
    -------
    tuple
        (word2id dictionary, embeddings array).
    """
    word2id = {}
    with open(wordpath, 'r') as f:
        for i, w in enumerate(list(f.readlines()[0].split())):
            word2id[w] = i
    embeddings = np.load(embpath)
    return word2id,embeddings


def df_to_t_features(df):
    """
    Convert DataFrame timestamps to time features.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'created_at' column containing timestamps.

    Returns
    -------
    numpy.ndarray
        Array of time features for each timestamp.
    """
    t_features = np.asarray([extract_time_feature(t_str) for t_str in df['created_at']])
    return t_features


def check_class_sizes(ground_truths, predictions):
    """
    Check sizes of predicted classes against ground truth classes.

    Parameters
    ----------
    ground_truths : array-like
        Ground truth class labels.
    predictions : array-like
        Predicted class labels.

    Returns
    -------
    list
        List of predicted class labels that are larger than average ground truth class size.
    """
    count_true_labels = list(Counter(ground_truths).values())  
    ave_true_size = mean(count_true_labels)
    distinct_predictions = list(Counter(predictions).keys()) 
    count_predictions = list(Counter(predictions).values()) 
    large_classes = [distinct_predictions[i] for i, count in enumerate(count_predictions) if count > ave_true_size]
    return large_classes

