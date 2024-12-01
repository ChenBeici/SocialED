import numpy as np
import json
import argparse
from torch.utils.data import Dataset
import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import combinations
import time
from time import localtime, strftime
import os
import pickle
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn import metrics
import sys
import pandas as pd
import en_core_web_lg
from datetime import datetime
import networkx as nx
from dgl.dataloading import MultiLayerNeighborSampler, NodeDataLoader




"""
    KPGNN Model
    Paper: Knowledge-Preserving Incremental Social Event Detection via Heterogeneous GNNs
    Author: Yuwei Cao et al.
    github: https://github.com/RingBDStack/KPGNN
"""

class args_define():
    
    parser = argparse.ArgumentParser()
    # Hyper-parameters
    parser.add_argument('--n_epochs', default=15, type=int,
                        help="Number of initial-training/maintenance-training epochs.")
    parser.add_argument('--n_infer_epochs', default=0, type=int,
                        help="Number of inference epochs.")
    parser.add_argument('--window_size', default=3, type=int,
                        help="Maintain the model after predicting window_size blocks.")
    parser.add_argument('--patience', default=5, type=int,
                        help="Early stop if performance did not improve in the last patience epochs.")
    parser.add_argument('--margin', default=3., type=float,
                        help="Margin for computing triplet losses")
    parser.add_argument('--lr', default=1e-3, type=float,
                        help="Learning rate")
    parser.add_argument('--batch_size', default=200, type=int,
                        help="Batch size (number of nodes sampled to compute triplet loss in each batch)")
    parser.add_argument('--n_neighbors', default=800, type=int,
                        help="Number of neighbors sampled for each node.")
    parser.add_argument('--hidden_dim', default=8, type=int,
                        help="Hidden dimension")
    parser.add_argument('--out_dim', default=32, type=int,
                        help="Output dimension of tweet representations")
    parser.add_argument('--num_heads', default=4, type=int,
                        help="Number of heads in each GAT layer")
    parser.add_argument('--use_residual', dest='use_residual', default=True,
                        action='store_false',
                        help="If true, add residual(skip) connections")
    parser.add_argument('--validation_percent', default=0.2, type=float,
                        help="Percentage of validation nodes(tweets)")
    parser.add_argument('--use_hardest_neg', dest='use_hardest_neg', default=False,
                        action='store_true',
                        help="If true, use hardest negative messages to form triplets. Otherwise use random ones")
    parser.add_argument('--use_dgi', dest='use_dgi', default=False,
                        action='store_true',
                        help="If true, add a DGI term to the loss. Otherwise use triplet loss only")
    parser.add_argument('--remove_obsolete', default=2, type=int,
                        help="If 0, adopt the All Message Strategy: keep inseting new message blocks and never remove;\n" +
                             "if 1, adopt the Relevant Message Strategy: remove obsolete training nodes (that are not connected to the new messages arrived during the last window) during maintenance;\n" +
                             "if 2, adopt the Latest Message Strategy: during each prediction, use only the new data to construct message graph, during each maintenance, use only the last message block arrived during the last window for continue training.\n" +
                             "See Section 4.4 of the paper for detailed explanations of these strategies. Note the message graphs for 0/1 and 2 need to be constructed differently, see the head comment of custom_message_graph.py")
    parser.add_argument('--is_incremental', action='store_true', default=False,
                        help="static or incremental")
    # Other arguments
    parser.add_argument('--use_cuda', dest='use_cuda', default=False,
                        action='store_true',
                        help="Use cuda")
    parser.add_argument('--data_path', default='../model_saved/kpgnn/kpgnn_incremental_test', #default='./incremental_0808/',
                        type=str, help="Path of features, labels and edges")
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
    parser.add_argument('--mask_path', default=None,
                        type=str, help="File path that contains the training, validation and test masks")
    # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
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



class KPGNN():
    def __init__(self, args,dataset):
        self.args = args
        self.dataset = dataset
        self.model = None
        self.loss_fn = None
        self.loss_fn_dgi = None
        self.metrics = None
        self.train_indices = None
        self.indices_to_remove = None 
        self.embedding_save_path = None 
        self.data_split = None

    def preprocess(self):
        preprocessor = Preprocessor(self.args, self.dataset)
        #preprocessor.generate_initial_features(self.dataset)
        preprocessor.custom_message_graph(self.dataset)

    def fit(self):        
        use_cuda = self.args.use_cuda and torch.cuda.is_available()
        print("Using CUDA:", use_cuda)
        os.makedirs(self.args.data_path, exist_ok=True)
        
        # make dirs and save args
        if self.args.resume_path is None:  # build a new dir if training from scratch
            self.embedding_save_path = self.args.data_path + '/embeddings_' + strftime("%m%d%H%M%S", localtime())
            os.mkdir(self.embedding_save_path)

        # resume training using original dir
        else:  
            self.embedding_save_path = self.args.resume_path
        print("embedding_save_path: ", self.embedding_save_path)

        with open(self.embedding_save_path + '/args.txt', 'w') as f:
            json.dump(self.args.__dict__, f, indent=2)

        # Load data splits
        self.data_split = np.load(self.args.data_path + '/data_split.npy')

        # Loss
        if self.args.use_hardest_neg:
            self.loss_fn = OnlineTripletLoss(self.args.margin, HardestNegativeTripletSelector(self.args.margin))
        else:
            self.loss_fn = OnlineTripletLoss(self.args.margin, RandomNegativeTripletSelector(self.args.margin))
        if self.args.use_dgi:
            self.loss_fn_dgi = torch.nn.BCEWithLogitsLoss()

        self.metrics = [AverageNonzeroTripletsMetric()]

        train_i = 0
        

        if ((self.args.resume_path is not None) and (self.args.resume_point == 0) and (self.args.resume_current)) or self.args.resume_path is None:
            if not self.args.use_dgi:
                self.train_indices, self.indices_to_remove, self.model = KPGNN_model.initial_maintain(train_i, 0, self.data_split, self.metrics, self.embedding_save_path, self.loss_fn, self.model)
            else:
                self.train_indices, self.indices_to_remove, self.model = KPGNN_model.initial_maintain(train_i, 0, self.data_split, self.metrics, self.embedding_save_path, self.loss_fn, None, self.loss_fn_dgi)

        
    def detection(self):
        
        train_i = 0
        if self.args.is_incremental:

            # Initialize the model, train_indices and indices_to_remove to avoid errors
            if self.args.resume_path is not None:
                self.model = None
                self.train_indices = None
                self.indices_to_remove = []

            # iterate through all blocks
            for i in range(1, self.data_split.shape[0]):
                # Inference (prediction)
                # Resume model from the previous, i.e., (i-1)th block or continue the new experiment. Otherwise (to resume from other blocks) skip this step.
                if ((self.args.resume_path is not None) and (self.args.resume_point == i - 1) and (not self.args.resume_current)) or self.args.resume_path is None:
                    if not self.args.use_dgi:
                        self.model = KPGNN_model.infer(train_i, i, self.data_split, self.metrics, self.embedding_save_path, self.loss_fn, self.train_indices, self.model, None,
                                    self.indices_to_remove)
                    else:
                        self.model = KPGNN_model.infer(train_i, i, self.data_split, self.metrics, self.embedding_save_path, self.loss_fn, self.train_indices, self.model,
                                    self.loss_fn_dgi, self.indices_to_remove)
                # Maintain
                # Resume model from the current, i.e., ith block or continue the new experiment. Otherwise (to resume from other blocks) skip this step.
                if ((self.args.resume_path is not None) and (self.args.resume_point == i) and (self.args.resume_current)) or self.args.resume_path is None:
                    if i % self.args.window_size == 0:
                        train_i = i
                        if not self.args.use_dgi:
                            self.train_indices, self.indices_to_remove, self.model = KPGNN_model.initial_maintain(train_i, i, self.data_split, self.metrics,
                                                                                    self.embedding_save_path, self.loss_fn, self.model)
                        else:
                            self.train_indices, self.indices_to_remove, self.model = KPGNN_model.initial_maintain(train_i, i, self.data_split, self.metrics,
                                                                                    self.embedding_save_path, self.loss_fn, self.model,
                                                                                    self.loss_fn_dgi)

        data = SocialDataset(self.args.data_path, 0)
        g = dgl.DGLGraph(data.matrix) 
        g.readonly()
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        
        predictions = []
        ground_truths = []
        self.detection_path = '../model_saved/kpgnn/detection_split/'
        os.makedirs(self.detection_path, exist_ok=True)

        test_indices = generateMasks(len(labels), self.data_split, 1, 0, 0.2, self.detection_path, num_indices_to_remove=0)

        g.ndata['features'] = features

        _, extract_features, extract_labels = extract_embeddings(g, self.model, len(labels), labels)
        
        # Extract labels
        test_indices = torch.load(self.detection_path + '/test_indices.pt')

        labels_true = extract_labels[test_indices]
        # Extract features
        X = extract_features[test_indices, :]
        assert labels_true.shape[0] == X.shape[0]
        n_test_tweets = X.shape[0]

        # Get the total number of classes
        n_classes = len(set(list(labels_true)))

        # kmeans clustering
        kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
        predictions = kmeans.labels_
        ground_truths = labels_true

        return predictions, ground_truths

    def evaluate(self, predictions, ground_truths):
        ars = metrics.adjusted_rand_score(ground_truths, predictions)

        # Calculate Adjusted Mutual Information (AMI)
        ami = metrics.adjusted_mutual_info_score(ground_truths, predictions)

        # Calculate Normalized Mutual Information (NMI)
        nmi = metrics.normalized_mutual_info_score(ground_truths, predictions)
        
        print(f"Model Adjusted Rand Index (ARI): {ars}")
        print(f"Model Adjusted Mutual Information (AMI): {ami}")
        print(f"Model Normalized Mutual Information (NMI): {nmi}")
        return ars, ami, nmi


class Preprocessor:
    def __init__(self,args,dataset):
        pass
    #generate_initial_features
    def generate_initial_features(self, dataset):
        save_path = './data/Event2012/kpgnn/'
        df = dataset
        
        os.makedirs(save_path, exist_ok=True)
        print("Data converted to dataframe.")
        print(df.shape)
        print(df.head(10))

        d_features = self.documents_to_features(df)
        print("Document features generated.")
        t_features = self.df_to_t_features(df)
        print("Time features generated.")
        combined_features = np.concatenate((d_features, t_features), axis=1)
        print("Concatenated document features and time features.")

        np.save(save_path + 'features_69612_0709_spacy_lg_zero_multiclasses_filtered.npy', combined_features)
        print("Initial features saved.")

    def documents_to_features(self, df):
        nlp = en_core_web_lg.load()
        print("df.filtered_words.head(10)", df.filtered_words.head(10))
        features = df.filtered_words.apply(lambda x: nlp(' '.join(x)).vector).values
        print("features.head(10)", features,"\n", "np.stack(features, axis=0)", np.stack(features, axis=0))
        return np.stack(features, axis=0)

    def extract_time_feature(self, t_str):
        t = datetime.fromisoformat(str(t_str))
        OLE_TIME_ZERO = datetime(1899, 12, 30)
        delta = t - OLE_TIME_ZERO
        return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]  # 86,400 seconds in day

    # encode the times-tamps of all the messages in the dataframe
    def df_to_t_features(self, df):
        t_features = np.asarray([self.extract_time_feature(t_str) for t_str in df['created_at']])
        return t_features
 
    #custom_message_graph  
    def custom_message_graph(self, dataset):
        save_path = '../model_saved/kpgnn/kpgnn_incremental_test/'
        '''
        if os.path.exists(save_path):
            pass
        else:
            os.mkdir(save_path)
        '''
        os.makedirs(save_path, exist_ok=True)

        df = dataset
        print("Data loaded.")

        # sort data by time
        df = df.sort_values(by='created_at').reset_index()

        # append date
        df['date'] = [d.date() for d in df['created_at']]

        # load features
        # the dimension of feature is 300 in this dataset
        f = np.load('./data/Event2012/kpgnn/features_69612_0709_spacy_lg_zero_multiclasses_filtered.npy')

        # generate test graphs, features, and labels
        message, data_split, all_graph_mins = self.construct_incremental_dataset(df, save_path, f, True)
        with open(save_path + "node_edge_statistics.txt", "w") as text_file:
            text_file.write(message)
        np.save(save_path + 'data_split.npy', np.asarray(data_split))
        print("Data split: ", data_split)
        np.save(save_path + 'all_graph_mins.npy', np.asarray(all_graph_mins))
        print("Time sepnt on heterogeneous -> homogeneous graph conversions: ", all_graph_mins)

    def construct_graph_from_df(self, df, G=None):
        if G is None:
            G = nx.Graph()
        for _, row in df.iterrows():
            tid = 't_' + str(row['tweet_id'])
            G.add_node(tid)
            G.nodes[tid]['tweet_id'] = True  # right-hand side value is irrelevant for the lookup

            user_ids = row['user_mentions']
            user_ids.append(row['user_id'])
            user_ids = ['u_' + str(each) for each in user_ids]
            # print(user_ids)
            G.add_nodes_from(user_ids)
            for each in user_ids:
                G.nodes[each]['user_id'] = True

            entities = row['entities']
            # entities = ['e_' + each for each in entities]
            # print(entities)
            G.add_nodes_from(entities)
            for each in entities:
                G.nodes[each]['entity'] = True

            words = row['sampled_words']
            words = ['w_' + each for each in words]
            # print(words)
            G.add_nodes_from(words)
            for each in words:
                G.nodes[each]['word'] = True

            edges = []
            edges += [(tid, each) for each in user_ids]
            edges += [(tid, each) for each in entities]
            edges += [(tid, each) for each in words]
            G.add_edges_from(edges)

        return G

    def networkx_to_dgl_graph(self, G, save_path=None):
        message = ''
        print('Start converting heterogeneous networkx graph to homogeneous dgl graph.')
        message += 'Start converting heterogeneous networkx graph to homogeneous dgl graph.\n'
        all_start = time.time()

        print('\tGetting a list of all nodes ...')
        message += '\tGetting a list of all nodes ...\n'
        start = time.time()
        all_nodes = list(G.nodes)
        mins = (time.time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # print('All nodes: ', all_nodes)
        # print('Total number of nodes: ', len(all_nodes))

        print('\tGetting adjacency matrix ...')
        message += '\tGetting adjacency matrix ...\n'
        start = time.time()
        A = nx.to_numpy_array(G)  # Returns the graph adjacency matrix as a NumPy matrix.
        mins = (time.time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # compute commuting matrices
        print('\tGetting lists of nodes of various types ...')
        message += '\tGetting lists of nodes of various types ...\n'
        start = time.time()
        tid_nodes = list(nx.get_node_attributes(G, 'tweet_id').keys())
        userid_nodes = list(nx.get_node_attributes(G, 'user_id').keys())
        word_nodes = list(nx.get_node_attributes(G, 'word').keys())
        entity_nodes = list(nx.get_node_attributes(G, 'entity').keys())
        del G
        mins = (time.time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\tConverting node lists to index lists ...')
        message += '\tConverting node lists to index lists ...\n'
        start = time.time()
        #  find the index of target nodes in the list of all_nodes
        indices_tid = [all_nodes.index(x) for x in tid_nodes]
        indices_userid = [all_nodes.index(x) for x in userid_nodes]
        indices_word = [all_nodes.index(x) for x in word_nodes]
        indices_entity = [all_nodes.index(x) for x in entity_nodes]
        del tid_nodes
        del userid_nodes
        del word_nodes
        del entity_nodes
        mins = (time.time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # ----------------------tweet-user-tweet----------------------
        print('\tStart constructing tweet-user-tweet commuting matrix ...')
        print('\t\t\tStart constructing tweet-user matrix ...')
        message += '\tStart constructing tweet-user-tweet commuting matrix ...\n\t\t\tStart constructing tweet-user ' \
                'matrix ...\n '
        start = time.time()
        w_tid_userid = A[np.ix_(indices_tid, indices_userid)]
        #  return a N(indices_tid)*N(indices_userid) matrix, representing the weight of edges between tid and userid
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # convert to scipy sparse matrix
        print('\t\t\tConverting to sparse matrix ...')
        message += '\t\t\tConverting to sparse matrix ...\n'
        start = time.time()
        s_w_tid_userid = sparse.csr_matrix(w_tid_userid)  # matrix compression
        del w_tid_userid
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tTransposing ...')
        message += '\t\t\tTransposing ...\n'
        start = time.time()
        s_w_userid_tid = s_w_tid_userid.transpose()
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tCalculating tweet-user * user-tweet ...')
        message += '\t\t\tCalculating tweet-user * user-tweet ...\n'
        start = time.time()
        s_m_tid_userid_tid = s_w_tid_userid * s_w_userid_tid  # homogeneous message graph
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tSaving ...')
        message += '\t\t\tSaving ...\n'
        start = time.time()
        if save_path is not None:
            sparse.save_npz(save_path + "s_m_tid_userid_tid.npz", s_m_tid_userid_tid)
            print("Sparse binary userid commuting matrix saved.")
            del s_m_tid_userid_tid
        del s_w_tid_userid
        del s_w_userid_tid
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # ----------------------tweet-ent-tweet------------------------
        print('\tStart constructing tweet-ent-tweet commuting matrix ...')
        print('\t\t\tStart constructing tweet-ent matrix ...')
        message += '\tStart constructing tweet-ent-tweet commuting matrix ...\n\t\t\tStart constructing tweet-ent matrix ' \
                '...\n '
        start = time.time()
        w_tid_entity = A[np.ix_(indices_tid, indices_entity)]
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # convert to scipy sparse matrix
        print('\t\t\tConverting to sparse matrix ...')
        message += '\t\t\tConverting to sparse matrix ...\n'
        start = time.time()
        s_w_tid_entity = sparse.csr_matrix(w_tid_entity)
        del w_tid_entity
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tTransposing ...')
        message += '\t\t\tTransposing ...\n'
        start = time.time()
        s_w_entity_tid = s_w_tid_entity.transpose()
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tCalculating tweet-ent * ent-tweet ...')
        message += '\t\t\tCalculating tweet-ent * ent-tweet ...\n'
        start = time.time()
        s_m_tid_entity_tid = s_w_tid_entity * s_w_entity_tid
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tSaving ...')
        message += '\t\t\tSaving ...\n'
        start = time.time()
        if save_path is not None:
            sparse.save_npz(save_path + "s_m_tid_entity_tid.npz", s_m_tid_entity_tid)
            print("Sparse binary entity commuting matrix saved.")
            del s_m_tid_entity_tid
        del s_w_tid_entity
        del s_w_entity_tid
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # ----------------------tweet-word-tweet----------------------
        print('\tStart constructing tweet-word-tweet commuting matrix ...')
        print('\t\t\tStart constructing tweet-word matrix ...')
        message += '\tStart constructing tweet-word-tweet commuting matrix ...\n\t\t\tStart constructing tweet-word ' \
                'matrix ...\n '
        start = time.time()
        w_tid_word = A[np.ix_(indices_tid, indices_word)]
        del A
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # convert to scipy sparse matrix
        print('\t\t\tConverting to sparse matrix ...')
        message += '\t\t\tConverting to sparse matrix ...\n'
        start = time.time()
        s_w_tid_word = sparse.csr_matrix(w_tid_word)
        del w_tid_word
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tTransposing ...')
        message += '\t\t\tTransposing ...\n'
        start = time.time()
        s_w_word_tid = s_w_tid_word.transpose()
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tCalculating tweet-word * word-tweet ...')
        message += '\t\t\tCalculating tweet-word * word-tweet ...\n'
        start = time.time()
        s_m_tid_word_tid = s_w_tid_word * s_w_word_tid
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tSaving ...')
        message += '\t\t\tSaving ...\n'
        start = time.time()
        if save_path is not None:
            sparse.save_npz(save_path + "s_m_tid_word_tid.npz", s_m_tid_word_tid)
            print("Sparse binary word commuting matrix saved.")
            del s_m_tid_word_tid
        del s_w_tid_word
        del s_w_word_tid
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # ----------------------compute tweet-tweet adjacency matrix----------------------
        print('\tComputing tweet-tweet adjacency matrix ...')
        message += '\tComputing tweet-tweet adjacency matrix ...\n'
        start = time.time()
        if save_path is not None:
            s_m_tid_userid_tid = sparse.load_npz(save_path + "s_m_tid_userid_tid.npz")
            print("Sparse binary userid commuting matrix loaded.")
            s_m_tid_entity_tid = sparse.load_npz(save_path + "s_m_tid_entity_tid.npz")
            print("Sparse binary entity commuting matrix loaded.")
            s_m_tid_word_tid = sparse.load_npz(save_path + "s_m_tid_word_tid.npz")
            print("Sparse binary word commuting matrix loaded.")

        s_A_tid_tid = s_m_tid_userid_tid + s_m_tid_entity_tid
        del s_m_tid_userid_tid
        del s_m_tid_entity_tid
        s_bool_A_tid_tid = (s_A_tid_tid + s_m_tid_word_tid).astype('bool')  # confirm the connect between tweets
        del s_m_tid_word_tid
        del s_A_tid_tid
        mins = (time.time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'
        all_mins = (time.time() - all_start) / 60
        print('\tOver all time elapsed: ', all_mins, ' mins\n')
        message += '\tOver all time elapsed: '
        message += str(all_mins)
        message += ' mins\n'

        if save_path is not None:
            sparse.save_npz(save_path + "s_bool_A_tid_tid.npz", s_bool_A_tid_tid)
            print("Sparse binary adjacency matrix saved.")
            s_bool_A_tid_tid = sparse.load_npz(save_path + "s_bool_A_tid_tid.npz")
            print("Sparse binary adjacency matrix loaded.")

        # create corresponding dgl graph
        G = dgl.DGLGraph(s_bool_A_tid_tid)
        print('We have %d nodes.' % G.number_of_nodes())
        print('We have %d edges.' % G.number_of_edges())
        print()
        message += 'We have '
        message += str(G.number_of_nodes())
        message += ' nodes.'
        message += 'We have '
        message += str(G.number_of_edges())
        message += ' edges.\n'

        return all_mins, message

    def construct_incremental_dataset(self, df, save_path, features, test=True):
        # If test equals true, construct the initial graph using test_ini_size tweets
        # and increment the graph by test_incr_size tweets each day
        test_ini_size = 500
        test_incr_size = 100

        # save data splits for training/validate/test mask generation
        data_split = []
        # save time spent for the heterogeneous -> homogeneous conversion of each graph
        all_graph_mins = []
        message = ""
        # extract distinct dates
        distinct_dates = df.date.unique()  # 2012-11-07
        # print("Distinct dates: ", distinct_dates)
        print("Number of distinct dates: ", len(distinct_dates))
        print()
        message += "Number of distinct dates: "
        message += str(len(distinct_dates))
        message += "\n"

        # split data by dates and construct graphs
        # first week -> initial graph (20254 tweets)
        print("Start constructing initial graph ...")
        message += "\nStart constructing initial graph ...\n"
        ini_df = df.loc[df['date'].isin(distinct_dates[:7])]  # find top 7 dates
        if test:
            ini_df = ini_df[:test_ini_size]  # top test_ini_size dates
        G = self.construct_graph_from_df(ini_df)
        path = save_path + '0/'
        if os.path.exists(path):
            pass
        else:
            os.mkdir(path)
        grap_mins, graph_message = self.networkx_to_dgl_graph(G, save_path=path)
        message += graph_message
        print("Initial graph saved")
        message += "Initial graph saved\n"
        # record the total number of tweets
        data_split.append(ini_df.shape[0])
        # record the time spent for graph conversion
        all_graph_mins.append(grap_mins)
        # extract and save the labels of corresponding tweets
        y = ini_df['event_id'].values
        y = [int(each) for each in y]
        np.save(path + 'labels.npy', np.asarray(y))
        print("Labels saved.")
        message += "Labels saved.\n"
        # extract and save the features of corresponding tweets
        indices = ini_df['index'].values.tolist()
        x = features[indices, :]
        np.save(path + 'features.npy', x)
        print("Features saved.")
        message += "Features saved.\n\n"

        # subsequent days -> insert tweets day by day (skip the last day because it only contains one tweet)
        for i in range(7, len(distinct_dates) - 1):
            print("Start constructing graph ", str(i - 6), " ...")
            message += "\nStart constructing graph "
            message += str(i - 6)
            message += " ...\n"
            incr_df = df.loc[df['date'] == distinct_dates[i]]
            if test:
                incr_df = incr_df[:test_incr_size]

            # All/Relevant Message Strategy: keeping all the messages when constructing the graphs 
            # (for the Relevant Message Strategy, the unrelated messages will be removed from the graph later on).
            # G = construct_graph_from_df(incr_df, G) 

            # Latest Message Strategy: construct graph using only the data of the day
            G = self.construct_graph_from_df(incr_df)  

            path = save_path + str(i - 6) + '/'
            if os.path.exists(path):
                pass
            else:
                os.mkdir(path)

            grap_mins, graph_message = self.networkx_to_dgl_graph(G, save_path=path)
            message += graph_message
            print("Graph ", str(i - 6), " saved")
            message += "Graph "
            message += str(i - 6)
            message += " saved\n"
            # record the total number of tweets
            data_split.append(incr_df.shape[0])
            # record the time spent for graph conversion
            all_graph_mins.append(grap_mins)
            # extract and save the labels of corresponding tweets
            # y = np.concatenate([y, incr_df['event_id'].values], axis = 0)
            y = [int(each) for each in incr_df['event_id'].values]
            np.save(path + 'labels.npy', y)
            print("Labels saved.")
            message += "Labels saved.\n"
            # extract and save the features of corresponding tweets
            indices = incr_df['index'].values.tolist()
            x = features[indices, :]
            # x = np.concatenate([x, x_incr], axis = 0)
            np.save(path + 'features.npy', x)
            print("Features saved.")
            message += "Features saved.\n"

        return message, data_split, all_graph_mins


class KPGNN_model():
    def __init__(self):
        pass

    # Inference(prediction)
    def infer(train_i, i, data_split, metrics, embedding_save_path, loss_fn, train_indices=None, model=None,
            loss_fn_dgi=None, indices_to_remove=[]):
        args = args_define.args

        save_path_i = embedding_save_path + '/block_' + str(i)
        if not os.path.isdir(save_path_i):
            os.mkdir(save_path_i)

        data = SocialDataset(args.data_path, i)
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        print("labels1:", labels)
        in_feats = features.shape[1]  

        g = dgl.graph((data.matrix.row, data.matrix.col))  
        num_isolated_nodes = graph_statistics(g, save_path_i)

        if args.remove_obsolete == 1:
            if ((args.resume_path is not None) and (not args.resume_current) and (i == args.resume_point + 1) and (i > args.window_size)) \
                    or (indices_to_remove == [] and i > args.window_size):
                temp_i = max(((i - 1) // args.window_size) * args.window_size, 0)
                indices_to_remove = np.load(embedding_save_path + '/block_' + str(temp_i) + '/indices_to_remove.npy').tolist()

            if indices_to_remove != []:
                data.remove_obsolete_nodes(indices_to_remove)
                features = torch.FloatTensor(data.features)
                labels = torch.LongTensor(data.labels)
                print("labels2:", labels)
                g = dgl.graph((data.matrix.row, data.matrix.col))
                num_isolated_nodes = graph_statistics(g, save_path_i)

        if args.mask_path is None:
            mask_path = save_path_i + '/masks'
            if not os.path.isdir(mask_path):
                os.mkdir(mask_path)
            test_indices = generateMasks(len(labels), data_split, train_i, i, args.validation_percent, mask_path, len(indices_to_remove))
        else:
            test_indices = torch.load(args.mask_path + '/block_' + str(i) + '/masks/test_indices.pt')

        if args.use_cuda:
            features, labels = features.cuda(), labels.cuda()
            print("labels3:", labels)
            test_indices = test_indices.cuda()

        g.ndata['features'] = features

        if (args.resume_path is not None) and (not args.resume_current) and (i == args.resume_point + 1):
            if args.use_dgi:
                model = DGI(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
            else:
                model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)

            if args.use_cuda:
                model.cuda()

            model_path = embedding_save_path + '/block_' + str(args.resume_point) + '/models/best.pt'
            model.load_state_dict(torch.load(model_path))
            print("Resumed model from the previous block.")

            args.resume_path = None

        if train_indices is None:
            if args.remove_obsolete == 0 or args.remove_obsolete == 1:
                temp_i = max(((i - 1) // args.window_size) * args.window_size, 0)
                train_indices = torch.load(embedding_save_path + '/block_' + str(temp_i) + '/masks/train_indices.pt')
            else:
                if args.n_infer_epochs != 0:
                    print("==================================\n'continue training then predict' is unimplemented under remove_obsolete mode 2, will skip infer epochs.\n===================================\n")
                    args.n_infer_epochs = 0

        all_test_nmi = []
        time_predict = []

        message = "\n------------ Directly predict on block " + str(i) + " ------------\n"
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        start = time.time()

        extract_nids, extract_features, extract_labels = extract_embeddings(g, model, len(labels), labels)
        test_nmi = evaluate_model(extract_features, extract_labels, test_indices, -1, num_isolated_nodes, save_path_i, False)
        seconds_spent = time.time() - start
        message = '\nDirect prediction took {:.2f} seconds'.format(seconds_spent)
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        all_test_nmi.append(test_nmi)
        time_predict.append(seconds_spent)
        np.save(save_path_i + '/time_predict.npy', np.asarray(time_predict))

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        if args.n_infer_epochs != 0:
            message = "\n------------ Continue training then predict on block " + str(i) + " ------------\n"
            print(message)
            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)
        seconds_infer_batches = []
        mins_infer_epochs = []

        sampler = MultiLayerNeighborSampler([args.n_neighbors] * 2)
        dataloader = NodeDataLoader(
            g, train_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4)

        for epoch in range(args.n_infer_epochs):
            start_epoch = time.time()
            losses = []
            total_loss = 0
            if args.use_dgi:
                losses_triplet = []
                losses_dgi = []
            for metric in metrics:
                metric.reset()
            
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                start_batch = time.time()
                batch_features = blocks[0].srcdata['features']
                model.train()

                if args.use_dgi:
                    pred, ret = model(blocks, batch_features)
                else:
                    pred = model(blocks, batch_features)

                batch_labels = labels[output_nodes]
                loss_outputs = loss_fn(pred, batch_labels)
                loss = loss_outputs[0] if isinstance(loss_outputs, (tuple, list)) else loss_outputs
                if args.use_dgi:
                    n_samples = len(output_nodes)
                    lbl_1 = torch.ones(n_samples)
                    lbl_2 = torch.zeros(n_samples)
                    lbl = torch.cat((lbl_1, lbl_2), 0)
                    if args.use_cuda:
                        lbl = lbl.cuda()
                    losses_triplet.append(loss.item())
                    loss_dgi = loss_fn_dgi(ret, lbl)
                    losses_dgi.append(loss_dgi.item())
                    loss += loss_dgi
                    losses.append(loss.item())
                else:
                    losses.append(loss.item())
                total_loss += loss.item()

                for metric in metrics:
                    metric(pred, batch_labels, loss_outputs)

                if batch_id % args.log_interval == 0:
                    message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        batch_id * args.batch_size, train_indices.shape[0],
                        100. * batch_id / (train_indices.shape[0] // args.batch_size), np.mean(losses))
                    if args.use_dgi:
                        message += '\tLoss_triplet: {:.6f}'.format(np.mean(losses_triplet))
                        message += '\tLoss_dgi: {:.6f}'.format(np.mean(losses_dgi))
                    for metric in metrics:
                        message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
                    print(message)
                    with open(save_path_i + '/log.txt', 'a') as f:
                        f.write(message)
                    losses = []

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_seconds_spent = time.time() - start_batch
                seconds_infer_batches.append(batch_seconds_spent)

            total_loss /= (batch_id + 1)
            message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_infer_epochs, total_loss)
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            mins_spent = (time.time() - start_epoch) / 60
            message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
            message += '\n'
            print(message)
            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)
            mins_infer_epochs.append(mins_spent)

            # 验证
            extract_nids, extract_features, extract_labels = extract_embeddings(g, model, len(labels), labels)
            test_nmi = evaluate_model(extract_features, extract_labels, test_indices, epoch, num_isolated_nodes, save_path_i, False)
            all_test_nmi.append(test_nmi)
            # end one epoch

        # Save model (fine-tuned from the above continue training process)
        model_path = save_path_i + '/models'
        os.mkdir(model_path)
        p = model_path + '/best.pt'
        torch.save(model.state_dict(), p)
        print('Model saved.')

        # Save all test nmi
        np.save(save_path_i + '/all_test_nmi.npy', np.asarray(all_test_nmi))
        print('Saved all test nmi.')
        # Save time spent on epochs
        np.save(save_path_i + '/mins_infer_epochs.npy', np.asarray(mins_infer_epochs))
        print('Saved mins_infer_epochs.')
        # Save time spent on batches
        np.save(save_path_i + '/seconds_infer_batches.npy', np.asarray(seconds_infer_batches))
        print('Saved seconds_infer_batches.')

        return model


    # Train on initial/maintenance graphs, t == 0 or t % window_size == 0 in this paper
    def initial_maintain(train_i, i, data_split, metrics, embedding_save_path, loss_fn, model=None, loss_fn_dgi=None):
        args = args_define.args
        # make dir for graph i
        save_path_i = embedding_save_path + '/block_' + str(i)
        if not os.path.isdir(save_path_i):
            os.mkdir(save_path_i)

        # load data
        data = SocialDataset(args.data_path, i)
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        in_feats = features.shape[1]  # feature dimension

        # Construct graph that contains message blocks 0, ..., i if remove_obsolete = 0 or 1; graph that only contains message block i if remove_obsolete = 2
        g = dgl.DGLGraph(data.matrix) 
        num_isolated_nodes = graph_statistics(g, save_path_i)

        # if remove_obsolete is mode 1, resume or generate indices_to_remove, then remove obsolete nodes from the graph
        if args.remove_obsolete == 1:
            
            # Resume indices_to_remove from the current block
            if (args.resume_path is not None) and args.resume_current and (i == args.resume_point) and (i != 0):  
                indices_to_remove = np.load(save_path_i + '/indices_to_remove.npy').tolist()

            elif i == 0:  # generate empty indices_to_remove for initial block
                indices_to_remove = []
                # save indices_to_remove
                np.save(save_path_i + '/indices_to_remove.npy', np.asarray(indices_to_remove))

            #  update graph
            else:  # generate indices_to_remove for maintenance block
                # get the indices of all training nodes
                num_all_train_nodes = np.sum(data_split[:i + 1])
                all_train_indices = np.arange(0, num_all_train_nodes).tolist()
                # get the number of old training nodes added before this maintenance
                num_old_train_nodes = np.sum(data_split[:i + 1 - args.window_size])
                # indices_to_keep: indices of nodes that are connected to the new training nodes added at this maintenance
                # (include the indices of the new training nodes)
                indices_to_keep = list(set(data.matrix.indices[data.matrix.indptr[num_old_train_nodes]:]))
                # indices_to_remove is the difference between the indices of all training nodes and indices_to_keep
                indices_to_remove = list(set(all_train_indices) - set(indices_to_keep))
                # save indices_to_remove
                np.save(save_path_i + '/indices_to_remove.npy', np.asarray(indices_to_remove))

            if indices_to_remove != []:
                # remove obsolete nodes from the graph
                data.remove_obsolete_nodes(indices_to_remove)
                features = torch.FloatTensor(data.features)
                labels = torch.LongTensor(data.labels)
                # Reconstruct graph
                g = dgl.DGLGraph(data.matrix)  # graph that contains tweet blocks 0, ..., i
                num_isolated_nodes = graph_statistics(g, save_path_i)

        else:

            indices_to_remove = []

        # generate or load training/validate/test masks
        if (args.resume_path is not None) and args.resume_current and (
                i == args.resume_point):  # Resume masks from the current block

            train_indices = torch.load(save_path_i + '/masks/train_indices.pt')
            validation_indices = torch.load(save_path_i + '/masks/validation_indices.pt')
        if args.mask_path is None:

            mask_path = save_path_i + '/masks'
            if not os.path.isdir(mask_path):
                os.mkdir(mask_path)
            train_indices, validation_indices = generateMasks(len(labels), data_split, train_i, i, args.validation_percent,
                                                            mask_path, len(indices_to_remove))

        else:
            train_indices = torch.load(args.mask_path + '/block_' + str(i) + '/masks/train_indices.pt')
            validation_indices = torch.load(args.mask_path + '/block_' + str(i) + '/masks/validation_indices.pt')

        # Suppress warning
        g.set_n_initializer(dgl.init.zero_initializer)

        if args.use_cuda:
            features, labels = features.cuda(), labels.cuda()
            train_indices, validation_indices = train_indices.cuda(), validation_indices.cuda()

        g.ndata['features'] = features

        if (args.resume_path is not None) and args.resume_current and (
                i == args.resume_point):  # Resume model from the current block

            # Declare model
            if args.use_dgi:
                model = DGI(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
            else:
                model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)

            if args.use_cuda:
                model.cuda()

            # Load model from resume_point
            model_path = embedding_save_path + '/block_' + str(args.resume_point) + '/models/best.pt'
            model.load_state_dict(torch.load(model_path))
            print("Resumed model from the current block.")

            # Use resume_path as a flag
            args.resume_path = None

        elif model is None:  # Construct the initial model
            # Declare model
            if args.use_dgi:
                model = DGI(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
            else:
                model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)

            if args.use_cuda:
                model.cuda()

        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Start training
        message = "\n------------ Start initial training / maintaining using blocks 0 to " + str(i) + " ------------\n"
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        # record the highest validation nmi ever got for early stopping
        best_vali_nmi = 1e-9
        best_epoch = 0
        wait = 0
        # record validation nmi of all epochs before early stop
        all_vali_nmi = []
        # record the time spent in seconds on each batch of all training/maintaining epochs
        seconds_train_batches = []
        # record the time spent in mins on each epoch
        mins_train_epochs = []
        g.readonly()

        sampler = MultiLayerNeighborSampler([args.n_neighbors] * 2)  # args.n_hops 应该是 2

        # 创建 DataLoader
        dataloader = NodeDataLoader(
            g, train_indices, sampler,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=4)  # 设置为适当的 num_workers

        for epoch in range(args.n_epochs):
            start_epoch = time.time()
            model.train()
            total_loss = 0
            losses = []
            if args.use_dgi:
                losses_triplet = []
                losses_dgi = []

            for metric in metrics:
                metric.reset()

            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                start_batch = time.time()

                blocks = [block.int().to(torch.device('cuda' if args.use_cuda else 'cpu')) for block in blocks]
                batch_features = blocks[0].srcdata['features']
                batch_labels = labels[output_nodes]

                # forward
                if args.use_dgi:
                    pred, ret = model(blocks, batch_features)
                else:
                    blocks[0].srcdata['h'] = batch_features
                    #print(blocks[0].srcdata)

                    pred = model(blocks, batch_features)

                loss_outputs = loss_fn(pred, batch_labels)
                loss = loss_outputs[0] if isinstance(loss_outputs, (tuple, list)) else loss_outputs

                if args.use_dgi:
                    n_samples = len(output_nodes)
                    lbl_1 = torch.ones(n_samples)
                    lbl_2 = torch.zeros(n_samples)
                    lbl = torch.cat((lbl_1, lbl_2), 0)
                    if args.use_cuda:
                        lbl = lbl.cuda()
                    losses_triplet.append(loss.item())
                    loss_dgi = loss_fn_dgi(ret, lbl)
                    losses_dgi.append(loss_dgi.item())
                    loss += loss_dgi
                    losses.append(loss.item())
                else:
                    losses.append(loss.item())

                total_loss += loss.item()

                for metric in metrics:
                    metric(pred, batch_labels, loss_outputs)

                if batch_id % args.log_interval == 0:
                    message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        batch_id * args.batch_size, train_indices.shape[0],
                        100. * batch_id / (len(dataloader)), np.mean(losses))
                    if args.use_dgi:
                        message += '\tLoss_triplet: {:.6f}'.format(np.mean(losses_triplet))
                        message += '\tLoss_dgi: {:.6f}'.format(np.mean(losses_dgi))
                    for metric in metrics:
                        message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
                    print(message)
                    with open(save_path_i + '/log.txt', 'a') as f:
                        f.write(message)
                    losses = []

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_seconds_spent = time.time() - start_batch
                seconds_train_batches.append(batch_seconds_spent)
            
            
            total_loss /= (batch_id + 1)
            message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            mins_spent = (time.time() - start_epoch) / 60
            message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
            message += '\n'
            print(message)
            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)
            mins_train_epochs.append(mins_spent)

            # Validation
            # Infer the representations of all tweets
            g.readonly()
            extract_nids, extract_features, extract_labels = extract_embeddings(g, model, len(labels), labels)
            # Save the representations of all tweets
            # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
            # Evaluate the model: conduct kMeans clustering on the validation and report NMI
            validation_nmi = evaluate_model(extract_features, extract_labels, validation_indices, epoch, num_isolated_nodes,
                                    save_path_i, True)
            all_vali_nmi.append(validation_nmi)

            # Early stop
            if validation_nmi > best_vali_nmi:
                best_vali_nmi = validation_nmi
                best_epoch = epoch
                wait = 0
                # Save model
                model_path = save_path_i + '/models'
                if (epoch == 0) and (not os.path.isdir(model_path)):
                    os.mkdir(model_path)
                p = model_path + '/best.pt'
                torch.save(model.state_dict(), p)
                print('Best model saved after epoch ', str(epoch))
            else:
                wait += 1
            if wait == args.patience:
                print('Saved all_mins_spent')
                print('Early stopping at epoch ', str(epoch))
                print('Best model was at epoch ', str(best_epoch))
                break
            # end one epoch


        # Save all validation nmi
        np.save(save_path_i + '/all_vali_nmi.npy', np.asarray(all_vali_nmi))
        # Save time spent on epochs
        np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
        print('Saved mins_train_epochs.')
        # Save time spent on batches
        np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
        print('Saved seconds_train_batches.')

        # Load the best model of the current block
        best_model_path = save_path_i + '/models/best.pt'
        model.load_state_dict(torch.load(best_model_path))
        print("Best model loaded.")

        if args.remove_obsolete == 2:
            return None, indices_to_remove, model
        return train_indices, indices_to_remove, model



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

def generateMasks(length, data_split, train_i, i, validation_percent=0.2, save_path=None, num_indices_to_remove=0):
    """
        Intro:
        This function generates train and validation indices for initial/maintenance epochs and test indices for inference(prediction) epochs
        If remove_obsolete mode 0 or 1:
        For initial/maintenance epochs:
        - The first (train_i + 1) blocks (blocks 0, ..., train_i) are used as training set (with explicit labels)
        - Randomly sample validation_percent of the training indices as validation indices
        For inference(prediction) epochs:
        - The (i + 1)th block (block i) is used as test set
        Note that other blocks (block train_i + 1, ..., i - 1) are also in the graph (without explicit labels, only their features and structural info are leveraged)
        If remove_obsolete mode 2:
        For initial/maintenance epochs:
        - The (i + 1) = (train_i + 1)th block (block train_i = i) is used as training set (with explicit labels)
        - Randomly sample validation_percent of the training indices as validation indices
        For inference(prediction) epochs:
        - The (i + 1)th block (block i) is used as test set

        :param length: the length of label list
        :param data_split: loaded splited data (generated in custom_message_graph.py)
        :param train_i, i: flag, indicating for initial/maintenance stage if train_i == i and inference stage for others
        :param validation_percent: the percent of validation data occupied in whole dataset
        :param save_path: path to save data
        :param num_indices_to_remove: number of indices ought to be removed

        :returns train indices, validation indices or test indices
    """
    args = args_define.args
    if args.remove_obsolete == 0 or args.remove_obsolete == 1:  # remove_obsolete mode 0 or 1
        # verify total number of nodes
        assert length == (np.sum(data_split[:i + 1]) - num_indices_to_remove)

        # If is in initial/maintenance epochs, generate train and validation indices
        if train_i == i:
            # randomly shuffle the training indices
            train_length = np.sum(data_split[:train_i + 1])
            train_length -= num_indices_to_remove
            train_indices = torch.randperm(int(train_length))
            # get total number of validation indices
            n_validation_samples = int(train_length * validation_percent)
            # sample n_validation_samples validation indices and use the rest as training indices
            validation_indices = train_indices[:n_validation_samples]
            train_indices = train_indices[n_validation_samples:]
            if save_path is not None:
                torch.save(validation_indices, save_path + '/validation_indices.pt')
                torch.save(train_indices, save_path + '/train_indices.pt')
                validation_indices = torch.load(save_path + '/validation_indices.pt')
                train_indices = torch.load(save_path + '/train_indices.pt')
            return train_indices, validation_indices
        # If the process is in inference(prediction) epochs, generate test indices
        else:
            test_indices = torch.range(0, (data_split[i] - 1), dtype=torch.long)
            test_indices += (np.sum(data_split[:i]) - num_indices_to_remove)
            if save_path is not None:
                torch.save(test_indices, save_path + '/test_indices.pt')
                test_indices = torch.load(save_path + '/test_indices.pt')
            return test_indices

    else:  # remove_obsolete mode 2
        # verify total number of nodes
        assert length == data_split[i]

        # If is in initial/maintenance epochs, generate train and validation indices
        if train_i == i:
            # randomly shuffle the graph indices
            train_indices = torch.randperm(length)
            # get total number of validation indices
            n_validation_samples = int(length * validation_percent)
            # sample n_validation_samples validation indices and use the rest as training indices
            validation_indices = train_indices[:n_validation_samples]
            train_indices = train_indices[n_validation_samples:]
            if save_path is not None:
                torch.save(validation_indices, save_path +
                        '/validation_indices.pt')
                torch.save(train_indices, save_path + '/train_indices.pt')
                validation_indices = torch.load(
                    save_path + '/validation_indices.pt')
                train_indices = torch.load(save_path + '/train_indices.pt')
            return train_indices, validation_indices
        # If is in inference(prediction) epochs, generate test indices
        else:
            test_indices = torch.range(
                0, (data_split[i] - 1), dtype=torch.long)
            if save_path is not None:
                torch.save(test_indices, save_path + '/test_indices.pt')
                test_indices = torch.load(save_path + '/test_indices.pt')
            return test_indices

# Compute the representations of all the nodes in g using model
def extract_embeddings(g, model, num_all_samples, labels):
    args = args_define.args
    sampler = MultiLayerNeighborSampler([1000, 1000])  
    dataloader = NodeDataLoader(
        g, torch.arange(g.num_nodes()), sampler,
        batch_size=num_all_samples,  
        shuffle=False,
        drop_last=False,
        num_workers=4)  # 设置合适的 num_workers

    with torch.no_grad():
        model.eval()
        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            batch_features = blocks[0].srcdata['features']
            
            if args.use_dgi:
                extract_features, _ = model(blocks, batch_features)
            else:
                extract_features = model(blocks, batch_features)
                
            extract_nids = output_nodes.to(device=extract_features.device, dtype=torch.long)
            extract_labels = labels[extract_nids]

            assert batch_id == 0
            extract_nids = extract_nids.cpu().numpy()
            extract_features = extract_features.cpu().numpy()
            extract_labels = extract_labels.cpu().numpy()

        A = np.arange(num_all_samples)
        assert (A == extract_nids).all()

    return extract_nids, extract_features, extract_labels

def save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, path, counter):
    np.savetxt(path + '/features_' + str(counter) + '.tsv', extract_features, delimiter='\t')
    np.savetxt(path + '/labels_' + str(counter) + '.tsv', extract_labels, fmt='%i', delimiter='\t')
    with open(path + '/labels_tags_' + str(counter) + '.tsv', 'w') as f:
        f.write('label\tmessage_id\ttrain_tag\n')
        for (label, mid, train_tag) in zip(extract_labels, extract_nids, extract_train_tags):
            f.write("%s\t%s\t%s\n" % (label, mid, train_tag))
    print("Embeddings after inference epoch " + str(counter) + " saved.")
    print()

def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def run_kmeans(extract_features, extract_labels, indices, isoPath=None):
    # Extract the features and labels of the test tweets
    indices = indices.cpu().detach().numpy()

    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()
        non_isolated_index = list(np.where(temp != 1)[0])
        indices = intersection(indices, non_isolated_index)

    # Extract labels
    labels_true = extract_labels[indices]
    # Extract features
    X = extract_features[indices, :]
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]

    # Get the total number of classes     
    n_classes = len(set(list(labels_true))) 

    # k-means clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels = kmeans.labels_
    print("n_classes:", n_classes)
    print("labels_true list:", labels_true.tolist())
    print("labels_pred list:", labels.tolist())

    nmi = metrics.normalized_mutual_info_score(labels_true, labels)

    # Return number of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, nmi)

def evaluate_model(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path, is_validation=True):
    message = ''
    message += '\nEpoch '
    message += str(epoch)
    message += '\n'

    # with isolated nodes
    n_tweets, n_classes, nmi = run_kmeans(extract_features, extract_labels, indices)
    if is_validation:
        mode = 'validation'
    else:
        mode = 'test'
    message += '\tNumber of ' + mode + ' tweets: '
    message += str(n_tweets)
    message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
    message += str(n_classes)
    message += '\n\t' + mode + ' NMI: '
    message += str(nmi)
    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        n_tweets, n_classes, nmi = run_kmeans(extract_features, extract_labels, indices,
                                            save_path + '/isolated_nodes.pt')
        message += '\tNumber of ' + mode + ' tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
        message += str(n_classes)
        message += '\n\t' + mode + ' NMI: '
        message += str(nmi)
    message += '\n'

    with open(save_path + '/evaluate.txt', 'a') as f:
        f.write(message)
    print(message)

    return nmi


class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'

class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'

#layer
class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual=False):
        super(GATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.use_residual = use_residual
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # Edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # Message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # Reduce UDF for equation (3) & (4)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, block):
        h = block.srcdata['h']
        z = self.fc(h)
        block.srcdata['z'] = z
        block.dstdata['z'] = z[:block.num_dst_nodes()]  # 确保 dstdata['z'] 也被正确设置

        block.apply_edges(self.edge_attention)
        block.update_all(self.message_func, self.reduce_func)

        if self.use_residual:
            return z[:block.num_dst_nodes()] + block.dstdata['h']
        else:
            return block.dstdata['h']

class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', use_residual=False):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, use_residual))
        self.merge = merge

    def forward(self, block):
        head_outs = [attn_head(block) for attn_head in self.heads]
        if self.merge == 'cat':
            return torch.cat(head_outs, dim=1)
        else:
            return torch.mean(torch.stack(head_outs), dim=0)

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads, 'cat', use_residual)
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, 'cat', use_residual)

    def forward(self, blocks, features):
        blocks[0].srcdata['h'] = features
        
        h = self.layer1(blocks[0])
        h = F.elu(h)
        blocks[1].srcdata['h'] = h

        h = self.layer2(blocks[1])
        
        return h

# Applies an average on seq, of shape (nodes, features)
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq):
        return torch.mean(seq, 0)

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 0)
        c_x = c_x.expand_as(h_pl)
        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 1)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 1)
        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2
        logits = torch.cat((sc_1, sc_2), 0)
        # print("testing, shape of logits: ", logits.size())
        return logits

class DGI(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False):
        super(DGI, self).__init__()
        self.gat = GAT(in_dim, hidden_dim, out_dim, num_heads, use_residual)
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(out_dim)

    def forward(self, nf):
        h_1 = self.gat(nf, False)
        c = self.read(h_1)
        c = self.sigm(c)
        h_2 = self.gat(nf, True)
        ret = self.disc(c, h_1, h_2)
        return h_1, ret

    # Detach the return variables
    def embed(self, nf):
        h_1 = self.gat(nf, False)
        return h_1.detach()

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
                loss_values = ap_distance - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
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

#utils
# Dataset
class SocialDataset(Dataset):
    def __init__(self, path, index):
        self.features = np.load(path + '/' + str(index) + '/features.npy')
        temp = np.load(path + '/' + str(index) + '/labels.npy', allow_pickle=True)
        self.labels = np.asarray([int(each) for each in temp])
        self.matrix = self.load_adj_matrix(path, index)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

    def load_adj_matrix(self, path, index):
        s_bool_A_tid_tid = sparse.load_npz(path + '/' + str(index) + '/s_bool_A_tid_tid.npz')
        print("Sparse binary adjacency matrix loaded.")
        return s_bool_A_tid_tid

    # Used by remove_obsolete mode 1
    def remove_obsolete_nodes(self, indices_to_remove=None):  # indices_to_remove: list
        # torch.range(0, (self.labels.shape[0] - 1), dtype=torch.long)
        if indices_to_remove is not None:
            all_indices = np.arange(0, self.labels.shape[0]).tolist()
            indices_to_keep = list(set(all_indices) - set(indices_to_remove))
            self.features = self.features[indices_to_keep, :]
            self.labels = self.labels[indices_to_keep]
            self.matrix = self.matrix[indices_to_keep, :]  # keep row
            self.matrix = self.matrix[:, indices_to_keep]  # keep column
            #  remove nodes from matrix


if __name__ == '__main__':
    from data_sets import Event2012_Dataset, Event2018_Dataset, MAVEN_Dataset, Arabic_Dataset

    args = args_define.args
    dataset = Event2012_Dataset.load_data()
    kpgnn = KPGNN(args,dataset)

    kpgnn.preprocess()
    kpgnn.fit()
    predictions, ground_truths = kpgnn.detection()

    # Evaluate model
    kpgnn.evaluate(predictions, ground_truths)
