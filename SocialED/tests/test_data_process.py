import unittest
import numpy as np
import pandas as pd
import networkx as nx
import torch
from ..utils.data_process import (
    construct_graph,
    load_data,
    documents_to_features,
    extract_time_feature,
    check_class_sizes
)

class TestDataProcess(unittest.TestCase):
    def setUp(self):
        # Sample data for testing
        self.sample_df = pd.DataFrame({
            'tweet_id': [1, 2],
            'user_mentions': [[123, 456], [789]],
            'user_id': [111, 222],
            'entities': [['entity1', 'entity2'], ['entity3']],
            'sampled_words': [['word1', 'word2'], ['word3']]
        })
        
    def test_construct_graph(self):
        G = construct_graph(self.sample_df)
        self.assertIsInstance(G, nx.Graph)
        # Check if nodes were added correctly
        self.assertTrue('t_1' in G.nodes())
        self.assertTrue('u_111' in G.nodes())
        self.assertTrue('entity1' in G.nodes())
        self.assertTrue('w_word1' in G.nodes())

    def test_check_class_sizes(self):
        ground_truths = [0, 0, 1, 1, 1, 2, 2]
        predictions = [0, 0, 0, 1, 1, 2, 2]
        large_classes = check_class_sizes(ground_truths, predictions)
        self.assertIsInstance(large_classes, list)

    def test_extract_time_feature(self):
        time_str = "2023-01-01T12:00:00"
        features = extract_time_feature(time_str)
        self.assertEqual(len(features), 2)
        self.assertIsInstance(features[0], float)
        self.assertIsInstance(features[1], float)

if __name__ == '__main__':
    unittest.main()
