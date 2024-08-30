import numpy as np
import pandas as pd
import en_core_web_lg
from datetime import datetime
import torch
from typing import Any, Dict, List
import math
import os
import dgl
import dgl.function as fn
import gc
from itertools import combinations
from scipy import sparse
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from torch.utils.data import Dataset
from torch.functional import Tensor
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear, BatchNorm1d, Sequential, ModuleList, ReLU, Dropout
from torch_geometric.data import Data
from torch_geometric.loader import NeighborSampler
import random
import argparse
from time import localtime, strftime, time
import networkx as nx
import json
import torch.optim as optim


class Preprocessor:
    def __init__(self):
        pass
    
    def documents_to_features(self, df):
        self.nlp = en_core_web_lg.load()
        features = df.filtered_words.apply(lambda x: self.nlp(' '.join(x)).vector).values
        return np.stack(features, axis=0)

    def extract_time_feature(self, t_str):
        t = datetime.fromisoformat(str(t_str))
        OLE_TIME_ZERO = datetime(1899, 12, 30)
        delta = t - OLE_TIME_ZERO
        return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]  # 86,400 seconds in day

    def df_to_t_features(self, df):
        t_features = np.asarray([self.extract_time_feature(t_str) for t_str in df['created_at']])
        return t_features

    def generate_initial_features(self, df, save_path='../model_saved/finevent/'):
        os.makedirs(save_path, exist_ok=True)
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
        combined_features = np.load(save_path + 'features_69612_0709_spacy_lg_zero_multiclasses_filtered.npy')
        print("Initial features loaded.")
        print(combined_features.shape)

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
            G.add_nodes_from(user_ids)
            for each in user_ids:
                G.nodes[each]['user_id'] = True

            entities = row['entities']
            G.add_nodes_from(entities)
            for each in entities:
                G.nodes[each]['entity'] = True

            words = row['sampled_words']
            words = ['w_' + each for each in words]
            G.add_nodes_from(words)
            for each in words:
                G.nodes[each]['word'] = True

            edges = []
            edges += [(tid, each) for each in user_ids]
            edges += [(tid, each) for each in entities]
            edges += [(tid, each) for each in words]
            G.add_edges_from(edges)

        return G

    def construct_incremental_dataset(self, df, save_path, features, test=True):
        test_ini_size = 500
        test_incr_size = 100

        data_split = []
        all_graph_mins = []
        message = ""
        distinct_dates = df.date.unique()
        print("Number of distinct dates: ", len(distinct_dates))
        print()
        message += "Number of distinct dates: "
        message += str(len(distinct_dates))
        message += "\n"

        print("Start constructing initial graph ...")
        message += "\nStart constructing initial graph ...\n"
        ini_df = df
        G = self.construct_graph_from_df(ini_df)
        path = save_path + '0/'
        os.makedirs(path, exist_ok=True)
        grap_mins, graph_message = self.networkx_to_dgl_graph(G, save_path=path)
        message += graph_message
        print("Initial graph saved")
        message += "Initial graph saved\n"
        data_split.append(ini_df.shape[0])
        all_graph_mins.append(grap_mins)
        y = ini_df['event_id'].values
        y = [int(each) for each in y]
        np.save(path + 'labels.npy', np.asarray(y))
        print("Labels saved.")
        message += "Labels saved.\n"
        indices = ini_df['index'].values.tolist()
        x = features[indices, :]
        np.save(path + 'features.npy', x)
        print("Features saved.")
        message += "Features saved.\n\n"
        
        for i in range(7, len(distinct_dates) - 1):
            print("Start constructing graph ", str(i - 6), " ...")
            message += "\nStart constructing graph "
            message += str(i - 6)
            message += " ...\n"
            incr_df = df.loc[df['date'] == distinct_dates[i]]
            if test:
                incr_df = incr_df[:test_incr_size]
            G = self.construct_graph_from_df(incr_df)  
            path = save_path + str(i - 6) + '/'
            os.makedirs(path, exist_ok=True)
            grap_mins, graph_message = self.networkx_to_dgl_graph(G, save_path=path)
            message += graph_message
            print("Graph ", str(i - 6), " saved")
            message += "Graph "
            message += str(i - 6)
            message += " saved\n"
            all_graph_mins.append(grap_mins)
            y = [int(each) for each in incr_df['event_id'].values]
            np.save(path + 'labels.npy', y)
            print("Labels saved.")
            message += "Labels saved.\n"
            indices = incr_df['index'].values.tolist()
            x = features[indices, :]
            np.save(path + 'features.npy', x)
            print("Features saved.")
            message += "Features saved.\n"

        return message, data_split, all_graph_mins

    def construct_graph(self, df, save_path='../model_saved/finevent/incremental_test/'):
        os.makedirs(save_path, exist_ok=True)

        df = df.sort_values(by='created_at').reset_index()
        df['date'] = [d.date() for d in df['created_at']]
        f = np.load('../model_saved/finevent/features_69612_0709_spacy_lg_zero_multiclasses_filtered.npy')
        message, data_split, all_graph_mins = self.construct_incremental_dataset(df, save_path, f, True)
        with open(save_path + "node_edge_statistics.txt", "w") as text_file:
            text_file.write(message)
        np.save(save_path + 'data_split.npy', np.asarray(data_split))
        np.save(save_path + 'all_graph_mins.npy', np.asarray(all_graph_mins))
        print("Time spent on heterogeneous -> homogeneous graph conversions: ", all_graph_mins)

    def networkx_to_dgl_graph(self, G, save_path=None):
        message = ''
        print('Start converting heterogeneous networkx graph to homogeneous dgl graph.')
        message += 'Start converting heterogeneous networkx graph to homogeneous dgl graph.\n'
        all_start = time()

        print('\tGetting a list of all nodes ...')
        message += '\tGetting a list of all nodes ...\n'
        start = time()
        all_nodes = list(G.nodes)
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\tGetting adjacency matrix ...')
        message += '\tGetting adjacency matrix ...\n'
        start = time()
        A = nx.to_numpy_array(G)   # 使用稀疏矩阵
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\tGetting lists of nodes of various types ...')
        message += '\tGetting lists of nodes of various types ...\n'
        start = time()
        tid_nodes = list(nx.get_node_attributes(G, 'tweet_id').keys())
        userid_nodes = list(nx.get_node_attributes(G, 'user_id').keys())
        word_nodes = list(nx.get_node_attributes(G, 'word').keys())
        entity_nodes = list(nx.get_node_attributes(G, 'entity').keys())
        del G
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\tConverting node lists to index lists ...')
        message += '\tConverting node lists to index lists ...\n'
        start = time()
        indices_tid = [all_nodes.index(x) for x in tid_nodes]
        indices_userid = [all_nodes.index(x) for x in userid_nodes]
        indices_word = [all_nodes.index(x) for x in word_nodes]
        indices_entity = [all_nodes.index(x) for x in entity_nodes]
        del tid_nodes
        del userid_nodes
        del word_nodes
        del entity_nodes
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # ----------------------tweet-user-tweet----------------------
        print('\tStart constructing tweet-user-tweet commuting matrix ...')
        print('\t\t\tStart constructing tweet-user matrix ...')
        message += '\tStart constructing tweet-user-tweet commuting matrix ...\n\t\t\tStart constructing tweet-user matrix ...\n'
        start = time()
        w_tid_userid = A[indices_tid, :][:, indices_userid]
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # convert to scipy sparse matrix
        print('\t\t\tConverting to sparse matrix ...')
        message += '\t\t\tConverting to sparse matrix ...\n'
        start = time()
        s_w_tid_userid = csr_matrix(w_tid_userid)  # matrix compression
        del w_tid_userid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tTransposing ...')
        message += '\t\t\tTransposing ...\n'
        start = time()
        s_w_userid_tid = s_w_tid_userid.transpose()
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tCalculating tweet-user * user-tweet ...')
        message += '\t\t\tCalculating tweet-user * user-tweet ...\n'
        start = time()
        s_m_tid_userid_tid = s_w_tid_userid * s_w_userid_tid  # homogeneous message graph
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tSaving ...')
        message += '\t\t\tSaving ...\n'
        start = time()
        if save_path is not None:
            sparse.save_npz(save_path + "s_m_tid_userid_tid.npz", s_m_tid_userid_tid)
            print("Sparse binary userid commuting matrix saved.")
            del s_m_tid_userid_tid
        del s_w_tid_userid
        del s_w_userid_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # ----------------------tweet-ent-tweet------------------------
        print('\tStart constructing tweet-ent-tweet commuting matrix ...')
        print('\t\t\tStart constructing tweet-ent matrix ...')
        message += '\tStart constructing tweet-ent-tweet commuting matrix ...\n\t\t\tStart constructing tweet-ent matrix ...\n'
        start = time()
        w_tid_entity = A[indices_tid, :][:, indices_entity]
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # convert to scipy sparse matrix
        print('\t\t\tConverting to sparse matrix ...')
        message += '\t\t\tConverting to sparse matrix ...\n'
        start = time()
        s_w_tid_entity = csr_matrix(w_tid_entity)
        del w_tid_entity
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tTransposing ...')
        message += '\t\t\tTransposing ...\n'
        start = time()
        s_w_entity_tid = s_w_tid_entity.transpose()
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tCalculating tweet-ent * ent-tweet ...')
        message += '\t\t\tCalculating tweet-ent * ent-tweet ...\n'
        start = time()
        s_m_tid_entity_tid = s_w_tid_entity * s_w_entity_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tSaving ...')
        message += '\t\t\tSaving ...\n'
        start = time()
        if save_path is not None:
            sparse.save_npz(save_path + "s_m_tid_entity_tid.npz", s_m_tid_entity_tid)
            print("Sparse binary entity commuting matrix saved.")
            del s_m_tid_entity_tid
        del s_w_tid_entity
        del s_w_entity_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # ----------------------tweet-word-tweet----------------------
        print('\tStart constructing tweet-word-tweet commuting matrix ...')
        print('\t\t\tStart constructing tweet-word matrix ...')
        message += '\tStart constructing tweet-word-tweet commuting matrix ...\n\t\t\tStart constructing tweet-word matrix ...\n'
        start = time()
        w_tid_word = A[indices_tid, :][:, indices_word]
        del A
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # convert to scipy sparse matrix
        print('\t\t\tConverting to sparse matrix ...')
        message += '\t\t\tConverting to sparse matrix ...\n'
        start = time()
        s_w_tid_word = csr_matrix(w_tid_word)
        del w_tid_word
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tTransposing ...')
        message += '\t\t\tTransposing ...\n'
        start = time()
        s_w_word_tid = s_w_tid_word.transpose()
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tCalculating tweet-word * word-tweet ...')
        message += '\t\t\tCalculating tweet-word * word-tweet ...\n'
        start = time()
        s_m_tid_word_tid = s_w_tid_word * s_w_word_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        print('\t\t\tSaving ...')
        message += '\t\t\tSaving ...\n'
        start = time()
        if save_path is not None:
            sparse.save_npz(save_path + "s_m_tid_word_tid.npz", s_m_tid_word_tid)
            print("Sparse binary word commuting matrix saved.")
            del s_m_tid_word_tid
        del s_w_tid_word
        del s_w_word_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'

        # ----------------------compute tweet-tweet adjacency matrix----------------------
        print('\tComputing tweet-tweet adjacency matrix ...')
        message += '\tComputing tweet-tweet adjacency matrix ...\n'
        start = time()
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
        s_bool_A_tid_tid = (s_A_tid_tid + s_m_tid_word_tid).astype(bool)  # confirm the connect between tweets
        del s_m_tid_word_tid
        del s_A_tid_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: '
        message += str(mins)
        message += ' mins\n'
        all_mins = (time() - all_start) / 60
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
        G = dgl.from_scipy(s_bool_A_tid_tid)
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

    def save_edge_index(self, data_path='../model_saved/finevent/incremental_test'):
        relation_ids = ['entity', 'userid', 'word']
        for i in range(22):
            save_multi_relational_graph(data_path, relation_ids, [0, i])
            print('edge index saved')
        print('all edge index saved')

class FinEvent:
    def __init__(self, args, dataset):
        self.dataset = dataset
        pass

    def preprocess(self):
        preprocessor = Preprocessor()
        #preprocessor.generate_initial_features(self.dataset)
        preprocessor.construct_graph(self.dataset)
        preprocessor.save_edge_index()

    def fit(self):
        
        # check CUDA
        print('Using CUDA:', torch.cuda.is_available())

        # create working path
        embedding_save_path = args.data_path + '/embeddings'
        os.makedirs(embedding_save_path, exist_ok=True)
        print('embedding save path: ', embedding_save_path)

        # record hyper-parameters
        with open(embedding_save_path + '/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
        
        print('Batch Size:', args.batch_size)
        print('Intra Agg Mode:', args.is_shared)
        print('Inter Agg Mode:', args.inter_opt)
        print('Reserve node config?', args.is_initial)
        # load number of messages in each bl ocks
        # e.g. data_split = [  500  ,   100, ...,  100]
        #                    block_0  block_1    block_n
        data_split = np.load(args.data_path + '/data_split.npy')

        # define loss function
        # contrastive loss in our paper
        if args.use_hardest_neg:
            loss_fn = OnlineTripletLoss(args.margin, HardestNegativeTripletSelector(args.margin))
        else:
            loss_fn = OnlineTripletLoss(args.margin, RandomNegativeTripletSelector(args.margin))

        # define metrics
        BCL_metrics = [AverageNonzeroTripletsMetric()]

        # define detection stage
        Streaming = FinEvent_model(args)

        # pre-train stage: train on initial graph
        train_i = 0
        self.model, self.RL_thresholds = Streaming.initial_maintain(train_i=train_i,
                                                        i=0,
                                                        metrics=BCL_metrics,
                                                        embedding_save_path=embedding_save_path,
                                                        loss_fn=loss_fn,
                                                        model=None)

        # detection-maintenance stage: incremental training and detection
        for i in range(1, data_split.shape[0]):
            # infer every block
            self.model = Streaming.inference(train_i=train_i,
                                        i=i,
                                        metrics=BCL_metrics,
                                        embedding_save_path=embedding_save_path,
                                        loss_fn=loss_fn,
                                        model=self.model,
                                        RL_thresholds=self.RL_thresholds)
        
            # maintenance in window size and desert the last block
            if i % args.window_size == 0 and i != data_split.shape[0] - 1:
                train_i = i
                self.model, self.RL_thresholds = Streaming.initial_maintain(train_i=train_i,
                                                                i=i,
                                                                metrics=BCL_metrics,
                                                                embedding_save_path=embedding_save_path,
                                                                loss_fn=loss_fn,
                                                                model=None)

    def detection(self):
        """
        :param eval_data_path: Path to the detection data
        :param eval_metrics: List of detection metrics
        :param embedding_save_path: Path to save embeddings if needed
        :param best_model_path: Path to the best trained model
        :param loss_fn: Loss function used during detection
        :return: None
        """

        start_time = time()

        # Load detection data
        print("Loading detection data...")
        relation_ids = ['entity', 'userid', 'word']
        #homo_data = create_homodataset(args.data_path, [0, 0], args.validation_percent)
        homo_data = create_offline_homodataset(args.data_path, [0, 0])
        multi_r_data = create_multi_relational_graph(args.data_path, relation_ids, [0,0])
        print("detection data loaded. Time elapsed: {:.2f} seconds".format(time() - start_time))

        device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')

        # Load the best trained model
        print("Loading the best trained model...")
        best_model_path = args.data_path + 'embeddings/block_0/models/best.pt'
        feat_dim = homo_data.x.size(1)
        num_relations = len(multi_r_data)
        
        self.model = MarGNN((feat_dim, args.hidden_dim, args.out_dim, args.heads), 
                            num_relations=num_relations, inter_opt=args.inter_opt, is_shared=args.is_shared)

        state_dict = torch.load(best_model_path)
        self.model.load_state_dict(state_dict)
        self.model.to(device)  # 将模型移动到指定设备（如果使用GPU）

        # 设置模型为评估模式
        self.model.eval()
        print("Best model loaded and set to eval mode. Time elapsed: {:.2f} seconds".format(time() - start_time))

        RL_thresholds = torch.FloatTensor(args.threshold_start0)
        filtered_multi_r_data = torch.load('../model_saved/finevent/multi_remain_data.pt')

        # Sampling nodes
        print("Sampling nodes...")
        sampler = MySampler(args.sampler)
        test_num_samples = homo_data.test_mask.size(0)
        num_batches = int(test_num_samples / args.batch_size) + 1
        
        extract_features = []

        for batch in range(num_batches):
            print(f"Processing batch {batch+1}/{num_batches}...")
            i_start = args.batch_size * batch
            i_end = min((batch + 1) * args.batch_size, test_num_samples)
            batch_nodes = homo_data.test_mask[i_start:i_end]
            batch_labels = homo_data.y[batch_nodes]
            adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1], batch_size=args.batch_size)
        
            # Perform prediction
            with torch.no_grad():
                pred = self.model(homo_data.x, adjs, n_ids, device, RL_thresholds)
            
            extract_features.append(pred.cpu().detach())
            print(f"Batch {batch+1} processed.")
            
        extract_features = torch.cat(extract_features, dim=0)

        all_nodes = homo_data.test_mask
        ground_truths = homo_data.y[all_nodes]

        X = extract_features.cpu().detach().numpy()
        assert ground_truths.shape[0] == X.shape[0]

        # Get the total number of classes
        n_classes = len(set(ground_truths.tolist()))

        # k-means clustering
        print("Performing k-means clustering...")
        kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
        predictions = kmeans.labels_
        print("k-means clustering done. Time elapsed: {:.2f} seconds".format(time() - start_time))

        print("Detection complete. Total time elapsed: {:.2f} seconds".format(time() - start_time))
        return ground_truths, predictions

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

class FinEvent_model():
    def __init__(self, args) -> None:
        # register args
        self.args = args

    def inference(self,
                  train_i, i,
                  metrics,
                  embedding_save_path,
                  loss_fn,
                  model,
                  RL_thresholds=None,
                  loss_fn_dgi=None):

        model = MarGNN()
        # make dir for graph i
        # ./incremental_0808//embeddings_0403005348/block_xxx
        save_path_i = embedding_save_path + '/block_' + str(i)
        if not os.path.isdir(save_path_i):
            os.mkdir(save_path_i)

        # load data
        relation_ids: List[str] = ['entity', 'userid', 'word']
        homo_data = create_homodataset(self.args.data_path, [train_i, i], self.args.validation_percent)
        multi_r_data = create_multi_relational_graph(self.args.data_path, relation_ids, [train_i, i])


        print('embedding save path: ', embedding_save_path)
        num_relations = len(multi_r_data)

        device = torch.device('cuda:0' if torch.cuda.is_available() and self.args.use_cuda else 'cpu')

        # input dimension (300 in our paper)
        features = homo_data.x
        feat_dim = features.size(1)

        # prepare graph configs for node filtering
        if self.args.is_initial:
            print('prepare node configures...')
            pre_node_dist(multi_r_data, homo_data.x, save_path_i)
            filter_path = save_path_i
        else:
            filter_path = save_path_i

        if model is None:
            assert 'Cannot find pre-trained model'

        # directly predict
        message = "\n------------ Directly predict on block " + str(i) + " ------------\n"
        print(message)
        print('RL Threshold using in this block:', RL_thresholds)

        model.eval()

        test_indices, labels = homo_data.test_mask, homo_data.y
        test_num_samples = test_indices.size(0)

        sampler = MySampler(self.args.sampler)

        # filter neighbor in advance to fit with neighbor sampling
        filtered_multi_r_data = RL_neighbor_filter(multi_r_data, RL_thresholds,
                                                   filter_path) if RL_thresholds is not None and self.args.sampler == 'RL_sampler' else multi_r_data

        # batch testing
        extract_features = torch.FloatTensor([])
        num_batches = int(test_num_samples / self.args.batch_size) + 1
        with torch.no_grad():
            for batch in range(num_batches):

                start_batch = time()

                # split batch
                i_start = self.args.batch_size * batch
                i_end = min((batch + 1) * self.args.batch_size, test_num_samples)
                batch_nodes = test_indices[i_start:i_end]
                if not len(batch_nodes):
                    continue
                # sampling neighbors of batch nodes
                adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1],
                                             batch_size=self.args.batch_size)

                pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)

                batch_seconds_spent = time() - start_batch

                # for we haven't shuffle the test indices(see utils.py),
                # the output embeddings can be simply stacked together
                extract_features = torch.cat((extract_features, pred.cpu().detach()), dim=0)

                del pred
                gc.collect()

        extract_labels = labels.cpu().numpy()
        labels_true = extract_labels[test_indices]
        n_classes = len(set(labels_true.tolist()))
        # save_embeddings(extract_features, save_path_i)
        PPpath = '/home/lipu/smed/fin_data/french/fin_eva_2018'
        print(f"save Evaluate_datas{i} to {PPpath}", end='')
        Evaluate_datas = {'msg_feats': extract_features, 'msg_tags': labels_true, 'n_clust': n_classes}
        if not os.path.exists(PPpath):
            os.makedirs(PPpath,exist_ok=True)
        np.save(PPpath +f'evaluate_data_M{i}.npy', Evaluate_datas)
        print('done')

        nmi, ami, ari,  = evaluate_model(extract_features,
                                                      labels,
                                                      indices=test_indices,
                                                      epoch=-1,  # just for test
                                                      num_isolated_nodes=0,
                                                      save_path=save_path_i,
                                                      is_validation=False,
                                                      cluster_type=self.args.cluster_type,
                                                      )

        k_score = {"NMI": nmi, "AMI": ami, "ARI": ari}
        del homo_data, multi_r_data, features, filtered_multi_r_data
        torch.cuda.empty_cache()

        return model, k_score

    # train on initial/maintenance graphs, t == 0 or t % window_size == 0 in this paper

    def initial_maintain(self,
                         train_i, i,
                         metrics,
                         embedding_save_path,
                         loss_fn,
                         model=None,
                         loss_fn_dgi=None):
        """
        :param i:
        :param data_split:
        :param metrics:
        :param embedding_save_path:
        :param loss_fn:
        :param model:
        :param loss_fn_dgi:
        :return:
        """

        # make dir for graph i
        # ./incremental_0808//embeddings_0403005348/block_xxx
        save_path_i = embedding_save_path + '/block_' + str(i)
        if not os.path.isdir(save_path_i):
            os.mkdir(save_path_i)

        # load data
        relation_ids: List[str] = ['entity', 'userid', 'word']
        homo_data = create_homodataset(self.args.data_path, [train_i, i], self.args.validation_percent)
        multi_r_data = create_multi_relational_graph(self.args.data_path, relation_ids, [train_i, i])
        num_relations = len(multi_r_data)
        
        device = torch.device('cuda' if torch.cuda.is_available() and self.args.use_cuda else 'cpu')

        # input dimension (300 in our paper)
        num_dim = homo_data.x.size(0)
        feat_dim = homo_data.x.size(1)

        # prepare graph configs for node filtering
        if self.args.is_initial:
            print('prepare node configures...')
            #pre_node_dist(multi_r_data, homo_data.x, save_path_i)
            filter_path = save_path_i
        else:
            filter_path = self.args.data_path + str(i)

        if model is None: # pre-training stage in our paper
            # print('Pre-Train Stage...')
            model = MarGNN((feat_dim, self.args.hidden_dim, self.args.out_dim, self.args.heads), 
                            num_relations=num_relations, inter_opt=self.args.inter_opt, is_shared=self.args.is_shared)

        # define sampler
        sampler = MySampler(self.args.sampler)
        # load model to device
        model.to(device)
        
        # initialize RL thresholds
        # RL_threshold: [[.5], [.5], [.5]]
        RL_thresholds = torch.FloatTensor(self.args.threshold_start0)

        # define optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr, weight_decay=1e-4)

        # record training log
        message = "\n------------ Start initial training / maintaining using block " + str(i) + " ------------\n"
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

        # step13: start training
        for epoch in range(self.args.n_epochs):
            start_epoch = time()
            losses = []
            total_loss = 0.0
            
            for metric in metrics:
                metric.reset()

            # Multi-Agent
            
            # filter neighbor in advance to fit with neighbor sampling
            #filtered_multi_r_data = RL_neighbor_filter(multi_r_data, RL_thresholds, filter_path) if epoch >= self.args.RL_start0 and self.args.sampler == 'RL_sampler' else multi_r_data

            filtered_multi_r_data = torch.load(self.args.file_path +'multi_remain_data.pt')

            print(f"Epoch {epoch+1}/{self.args.n_epochs} - Starting training...")
            model.train()

            train_num_samples, valid_num_samples = homo_data.train_mask.size(0), homo_data.val_mask.size(0)
            all_num_samples = train_num_samples + valid_num_samples

            # batch training
            num_batches = int(train_num_samples / self.args.batch_size) + 1
            for batch in range(num_batches):
                start_batch = time()

                # split batch
                i_start = self.args.batch_size * batch
                i_end = min((batch + 1) * self.args.batch_size, train_num_samples)
                batch_nodes = homo_data.train_mask[i_start:i_end]
                batch_labels = homo_data.y[batch_nodes]

                print(f"Epoch {epoch+1}/{self.args.n_epochs} - Batch {batch+1}/{num_batches}: Processing nodes {i_start} to {i_end}...")

                # sampling neighbors of batch nodes
                adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1], batch_size=self.args.batch_size)
                optimizer.zero_grad()

                pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)
                loss_outputs = loss_fn(pred, batch_labels)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

                losses.append(loss.item())
                total_loss += loss.item()

                for metric in metrics:
                    metric(pred, batch_labels, loss_outputs)

                if batch % self.args.log_interval == 0:
                    message = 'Train: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(batch * self.args.batch_size, train_num_samples, 100. * batch / ((train_num_samples // self.args.batch_size) + 1), np.mean(losses))
                    
                    for metric in metrics:
                        message += '\t{}: {:.4f}'.format(metric.name(), metric.value())

                    print(message)  # 输出到控制台
                    with open(save_path_i + '/log.txt', 'a') as f:
                        f.write(message)
                    losses = []

                del pred, loss_outputs
                gc.collect()

                print(f"Epoch {epoch+1}/{self.args.n_epochs} - Batch {batch+1}/{num_batches}: Performing backward pass...")
                loss.backward()
                optimizer.step()

                batch_seconds_spent = time() - start_batch
                seconds_train_batches.append(batch_seconds_spent)

                del loss
                gc.collect()
            

            # step14: print loss
            total_loss /= (batch + 1)
            message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch+1, self.args.n_epochs, total_loss)
            for metric in metrics:
                message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
            mins_spent = (time() - start_epoch) / 60
            message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
            message += '\n'
            print(message)
            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)
            mins_train_epochs.append(mins_spent)

            # validation
            # infer the representations of all tweets
            model.eval()

            # we recommand to forward all nodes and select the validation indices instead
            extract_features = torch.FloatTensor([])

            num_batches = int(all_num_samples / self.args.batch_size) + 1
            
            # all mask are then splited into mini-batch in order
            all_mask = torch.arange(0, num_dim, dtype=torch.long)

            for batch in range(num_batches):
                start_batch = time()

                # split batch
                i_start = self.args.batch_size * batch
                i_end = min((batch + 1) * self.args.batch_size, all_num_samples)
                batch_nodes = all_mask[i_start:i_end]
                batch_labels = homo_data.y[batch_nodes]

                # sampling neighbors of batch nodes
                adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1], batch_size=self.args.batch_size)

                pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)

                extract_features = torch.cat((extract_features, pred.cpu().detach()), dim=0)

                del pred
                gc.collect()

            # save_embeddings(extract_features, save_path_i)
            epoch = epoch + 1
            # evaluate the model: conduct kMeans clustering on the validation and report NMI
            validation_nmi = evaluate_model(extract_features[homo_data.val_mask], 
                                      homo_data.y,
                                      indices=homo_data.val_mask, 
                                      epoch=epoch,
                                      num_isolated_nodes=0, 
                                      save_path=save_path_i, 
                                      is_validation=True, 
                                      cluster_type=self.args.cluster_type)
            all_vali_nmi.append(validation_nmi)

            # step16: early stop
            if validation_nmi > best_vali_nmi:
                best_vali_nmi = validation_nmi
                best_epoch = epoch
                wait = 0
                # save model
                model_path = save_path_i + '/models'
                if (epoch == 1) and (not os.path.isdir(model_path)):
                    os.mkdir(model_path)
                p = model_path + '/best.pt'
                torch.save(model.state_dict(), p)
                print('Best model saved after epoch ', str(epoch))
            else:
                wait += 1
            if wait >= self.args.patience:
                print('Saved all_mins_spent')
                print('Early stopping at epoch ', str(epoch))
                print('Best model was at epoch ', str(best_epoch))
                break
            # end one epoch

        # save all validation nmi
        np.save(save_path_i + '/all_vali_nmi.npy', np.asarray(all_vali_nmi))
        # save time spent on epochs
        np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
        print('Saved mins_train_epochs.')
        # save time spent on batches
        np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
        print('Saved seconds_train_batches.')

        # load the best model of the current block
        best_model_path = save_path_i + '/models/best.pt'
        model.load_state_dict(torch.load(best_model_path))
        print("Best model loaded.")

        del homo_data, multi_r_data
        torch.cuda.empty_cache()
        
        return model, RL_thresholds

#gen_dataset
def sparse_trans(datapath='incremental_test/0/s_m_tid_userid_tid.npz'):
    relation = sparse.load_npz(datapath)
    all_edge_index = torch.tensor([], dtype=int)
    for node in range(relation.shape[0]):
        neighbor = torch.IntTensor(relation[node].toarray()).squeeze()
        # del self_loop in advance
        neighbor[node] = 0
        neighbor_idx = neighbor.nonzero()
        neighbor_sum = neighbor_idx.size(0)
        loop = torch.tensor(node).repeat(neighbor_sum, 1)
        edge_index_i_j = torch.cat((loop, neighbor_idx), dim=1).t()
        # edge_index_j_i = torch.cat((neighbor_idx, loop), dim=1).t()
        self_loop = torch.tensor([[node],[node]])
        all_edge_index = torch.cat((all_edge_index, edge_index_i_j, self_loop), dim=1)
        del neighbor, neighbor_idx, loop, self_loop, edge_index_i_j
    return all_edge_index

def coo_trans(datapath = 'incremental_test/0/s_m_tid_userid_tid.npz'):
    relation:csr_matrix = sparse.load_npz(datapath)
    relation:coo_matrix = relation.tocoo()
    sparse_edge_index = torch.LongTensor([relation.row, relation.col])
    return sparse_edge_index

def create_dataset(loadpath, relation, mode):
    features = np.load(os.path.join(loadpath, str(mode[1]), 'features.npy'))
    features = torch.FloatTensor(features)
    print('features loaded')
    labels = np.load(os.path.join(loadpath, str(mode[1]), 'labels.npy'))
    print('labels loaded')
    labels = torch.LongTensor(labels)
    relation_edge_index = coo_trans(os.path.join(loadpath, str(mode[1]), 's_m_tid_%s_tid.npz' % relation))
    print('edge index loaded')
    data = Data(x=features, edge_index=relation_edge_index, y=labels)
    data_split = np.load(os.path.join(loadpath, 'data_split.npy'))
    train_i, i = mode[0], mode[1]
    if train_i == i:
        data.train_mask, data.val_mask = generateMasks(len(labels), data_split, train_i, i)
    else:
        data.test_mask = generateMasks(len(labels), data_split, train_i, i)

    return data

def create_homodataset(loadpath, mode, valid_percent=0.2):
    features = np.load(os.path.join(loadpath, str(mode[1]), 'features.npy'))
    features = torch.FloatTensor(features)
    print('features loaded')
    labels = np.load(os.path.join(loadpath, str(mode[1]), 'labels.npy'))
    print('labels loaded')
    labels = torch.LongTensor(labels)
    data = Data(x=features, edge_index=None, y=labels)
    data_split = np.load(os.path.join(loadpath, 'data_split.npy'))
    train_i, i = mode[0], mode[1]
    if train_i == i:
        data.train_mask, data.val_mask = generateMasks(len(labels), data_split, train_i, i, valid_percent)
    else:
        data.test_mask = generateMasks(len(labels), data_split, train_i, i)

    return data

def create_offline_homodataset(loadpath, mode):
    features = np.load(os.path.join(loadpath, str(mode[1]), 'features.npy'))
    features = torch.FloatTensor(features)
    print('features loaded')
    labels = np.load(os.path.join(loadpath, str(mode[1]), 'labels.npy'))
    print('labels loaded')
    labels = torch.LongTensor(labels)
    # relation_edge_index = sparse_trans(os.path.join(loadpath, str(mode[1]), 's_bool_A_tid_tid.npz'))
    # print('edge index loaded')
    data = Data(x=features, edge_index=None, y=labels)
    data.train_mask, data.val_mask, data.test_mask = gen_offline_masks(len(labels))

    return data

def create_multi_relational_graph(loadpath, relations, mode):

    # multi_relation_edge_index = [sparse_trans(os.path.join(loadpath, str(mode[1]), 's_m_tid_%s_tid.npz' % relation)) for relation in relations]
    multi_relation_edge_index = [torch.load(loadpath + '/' + str(mode[1]) + '/edge_index_%s.pt' % relation) for relation in relations]
    print('sparse trans...')
    print('edge index loaded')

    return multi_relation_edge_index
    
def save_multi_relational_graph(loadpath, relations, mode):

    for relation in relations:
        relation_edge_index = sparse_trans(os.path.join(loadpath, str(mode[1]), 's_m_tid_%s_tid.npz' % relation))
        print('%s have saved' % (os.path.join(loadpath, str(mode[1]), 's_m_tid_%s_tid.npz' % relation)))
        torch.save(relation_edge_index, loadpath + '/' + str(mode[1]) + '/edge_index_%s.pt' % relation)

#utils
def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def run_hdbscan(extract_features, extract_labels, indices,is_validation, isoPath=None,):

    #2018:min_cluster_size = 5, copy = True, alpha = 0.8
    #2012:min_cluster_size = 8

    indices = indices.cpu().detach().numpy()

    if isoPath is not None:
        # Remove isolated points
        temp = torch.load(isoPath)
        temp = temp.cpu().detach().numpy()
        non_isolated_index = list(np.where(temp != 1)[0])
        indices = intersection(indices, non_isolated_index)

    # Extract labels
    extract_labels = extract_labels.cpu().numpy()
    labels_true = extract_labels[indices]

    # Extract features
    # X = extract_features[indices, :]
    X = extract_features.cpu().detach().numpy()
    assert labels_true.shape[0] == X.shape[0]
    nmi,ami,ari = 0,0,0
    for eps in [0.2,0.3,0.5,0.7,1,1.2,1.5,1.7,2,2.2,2.5,2.7,3,3.2,3.5,3.7,4,4.2,4.5,4.7,5]:
        hdb = DBSCAN(eps=eps,min_samples=8)
        hdb.fit(X)

        labels = hdb.labels_
        _nmi = metrics.normalized_mutual_info_score(labels_true, labels)
        _ami = metrics.adjusted_mutual_info_score(labels_true, labels)
        _ari = metrics.adjusted_rand_score(labels_true, labels)
        print(f"_nmi:{_nmi}\t _ami:{_ami}\t _ari:{_ari}\n")
        if _nmi > nmi:
            nmi=_nmi
            ami = _ami
            ari = _ari


    return nmi,ami,ari

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
    extract_labels = extract_labels.cpu().numpy()
    labels_true = extract_labels[indices]

    # Extract features
    # X = extract_features[indices, :]
    X = extract_features.cpu().detach().numpy()
    assert labels_true.shape[0] == X.shape[0]
    n_test_tweets = X.shape[0]  # 100

    # Get the total number of classes
    n_classes = len(set(labels_true.tolist()))

    # k-means clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels = kmeans.labels_

    nmi = metrics.normalized_mutual_info_score(labels_true, labels)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels)
    ari = metrics.adjusted_rand_score(labels_true, labels)

    # Return number of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return n_test_tweets, n_classes, nmi, ami, ari

def evaluate_model(extract_features, extract_labels, indices,
             epoch, num_isolated_nodes, save_path, is_validation=True, 
             cluster_type='kmeans'):

    message = ''
    message += '\nEpoch '
    message += str(epoch)
    message += '\n'

    # with isolated nodes
    if cluster_type == 'kmeans':
        n_tweets, n_classes, nmi, ami, ari = run_kmeans(extract_features, extract_labels, indices)
    elif cluster_type == 'dbscan':
        pass

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
    message += '\n\t' + mode + ' AMI: '
    message += str(ami)
    message += '\n\t' + mode + ' ARI: '
    message += str(ari)
    if cluster_type == 'dbscan':
        message += '\n\t' + mode + ' best_eps: '
        # message += str(best_eps)
        message += '\n\t' + mode + ' best_min_Pts: '
        # message += str(best_min_Pts)

    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        n_tweets, n_classes, nmi, ami, ari = run_kmeans(extract_features, extract_labels, indices,
                                                        save_path + '/isolated_nodes.pt')
        message += '\tNumber of ' + mode + ' tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
        message += str(n_classes)
        message += '\n\t' + mode + ' NMI: '
        message += str(nmi)
        message += '\n\t' + mode + ' AMI: '
        message += str(ami)
        message += '\n\t' + mode + ' ARI: '
        message += str(ari)
    message += '\n'

    with open(save_path + '/evaluate.txt', 'a') as f:
        f.write(message)
    print(message)

    np.save(save_path + '/%s_metric.npy' % mode, np.asarray([nmi, ami, ari]))

    return nmi

def generateMasks(length, data_split, train_i, i, validation_percent=0.2, save_path=None, remove_obsolete=2):
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
    print(length)
    print(data_split[i])

    # step1: verify total number of nodes
    assert length == data_split[i]  # 500

       # step2.0: if is in initial/maintenance epochs, generate train and validation indices
    if train_i == i:
        # step3: randomly shuffle the graph indices
        train_indices = torch.randperm(length)

        # step4: get total number of validation indices
        n_validation_samples = int(length * validation_percent)

        # step5: sample n_validation_samples validation indices and use the rest as training indices
        validation_indices = train_indices[:n_validation_samples]
        train_indices = train_indices[n_validation_samples:]

        # print(save_path) #./incremental_0808//embeddings_0403100832/block_0/masks
        # step6: save indices
        if save_path is not None:
            torch.save(train_indices, save_path + '/train_indices.pt')
            torch.save(validation_indices, save_path + '/validation_indices.pt')
            
        return train_indices, validation_indices
        # step2.1: if is in inference(prediction) epochs, generate test indices
    else:
        test_indices = torch.arange(0, (data_split[i]), dtype=torch.long)
        if save_path is not None:
            torch.save(test_indices, save_path + '/test_indices.pt')

        return test_indices

def gen_offline_masks(length, validation_percent=0.2, test_percent=0.1):

    test_length = int(length * test_percent)
    valid_length = int(length * validation_percent)
    train_length = length - valid_length - test_length

    samples = torch.randperm(length)
    train_indices = samples[:train_length]
    valid_indices = samples[train_length:train_length + valid_length]
    test_indices = samples[train_length + valid_length:]


    return train_indices, valid_indices, test_indices

def save_embeddings(extracted_features, save_path):
    
    torch.save(extracted_features, save_path + '/final_embeddings.pt')
    print('extracted features saved.')

#Mysampler
class MySampler(object):

    def __init__(self, sampler) -> None:
        super().__init__()

        self.sampler = sampler

    def sample(self, multi_relational_edge_index: List[Tensor], node_idx, sizes, batch_size):
        
        if self.sampler == 'RL_sampler':
            return self._RL_sample(multi_relational_edge_index, node_idx, sizes, batch_size)
        elif self.sampler == 'random_sampler':
            return self._random_sample(multi_relational_edge_index, node_idx, batch_size)
        elif self.sampler == 'const_sampler':
            return self._const_sample(multi_relational_edge_index, node_idx, batch_size)

    def _RL_sample(self, multi_relational_edge_index: List[Tensor], node_idx, sizes, batch_size):

        outs = []
        all_n_ids = []
        for id, edge_index in enumerate(multi_relational_edge_index):
            loader = NeighborSampler(edge_index=edge_index, 
                                     sizes=sizes, 
                                     node_idx=node_idx,
                                     return_e_id=False,
                                     batch_size=batch_size,
                                     num_workers=0)
            for id, (_, n_ids, adjs) in enumerate(loader):
                # print(adjs)
                outs.append(adjs)
                all_n_ids.append(n_ids)

            #print(id)
            assert id == 0

        return outs, all_n_ids

    def _random_sample(self, multi_relational_edge_index: List[Tensor], node_idx, batch_size):

        outs = []
        all_n_ids = []

        sizes = [random.randint(10, 100), random.randint(10, 50)]
        for edge_index in multi_relational_edge_index:
            loader = NeighborSampler(edge_index=edge_index, 
                                    sizes=sizes, 
                                    node_idx=node_idx,
                                    return_e_id=False,
                                    batch_size=batch_size,
                                    num_workers=0)
            for id, (_, n_ids, adjs) in enumerate(loader):
                # print(adjs)
                outs.append(adjs)
                all_n_ids.append(n_ids)

            # print(id)
            assert id == 0

        return outs, all_n_ids

    def _const_sample(self, multi_relational_edge_index: List[Tensor], node_idx, batch_size):

        outs = []
        all_n_ids = []
        sizes = [25, 15]
        for edge_index in multi_relational_edge_index:

            loader = NeighborSampler(edge_index=edge_index, 
                                    sizes=sizes, 
                                    node_idx=node_idx,
                                    return_e_id=False,
                                    batch_size=batch_size,
                                    num_workers=0)
            for id, (_, n_ids, adjs) in enumerate(loader):
                # print(adjs)
                outs.append(adjs)
                all_n_ids.append(n_ids)

            # print(id)
            assert id == 0

        return outs, all_n_ids

#Metrics
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

#model
class MarGNN(nn.Module):
    def __init__(self, GNN_args, num_relations, inter_opt, is_shared=False):
        super(MarGNN, self).__init__()

        self.num_relations = num_relations
        self.inter_opt = inter_opt
        self.is_shared = is_shared
        if not self.is_shared:
            self.intra_aggs = torch.nn.ModuleList([Intra_AGG(GNN_args) for _ in range(self.num_relations)])
        else:
            self.intra_aggs = Intra_AGG(GNN_args) # shared parameters
        
        if self.inter_opt == 'cat_w_avg_mlp' or 'cat_wo_avg_mlp':
            in_dim, hid_dim, out_dim, heads = GNN_args
            mlp_args = self.num_relations * out_dim, out_dim
            self.inter_agg = Inter_AGG(mlp_args)
        else:
            self.inter_agg = Inter_AGG()

    def forward(self, x, adjs, n_ids, device, RL_thresholds):

        # RL_threshold: tensor([[.5], [.5], [.5]])
        if RL_thresholds is None:
            RL_thresholds = torch.FloatTensor([[1.], [1.], [1.]])
        if not isinstance(RL_thresholds, Tensor):
            RL_thresholds = torch.FloatTensor(RL_thresholds)

        RL_thresholds = RL_thresholds.to(device)

        features = []
        for i in range(self.num_relations):
            if not self.is_shared:
                # print('Intra Aggregation of relation %d' % i)
                features.append(self.intra_aggs[i](x[n_ids[i]], adjs[i], device))
            else:
                # shared parameters.
                # print('Shared Intra Aggregation...')
                features.append(self.intra_aggs(x[n_ids[i]], adjs[i], device))

        features = torch.stack(features, dim=0)
        
        features = self.inter_agg(features, RL_thresholds, self.inter_opt) 

        return features

#env
def RL_neighbor_filter_full(multi_r_data, RL_thresholds, features, save_path=None):

    multi_remain_data = []
    multi_r_score = []

    for i, r_data in enumerate(multi_r_data):
        r_data: Tensor
        unique_nodes = r_data[1].unique()
        num_nodes = unique_nodes.size(0)
        remain_node_index = torch.tensor([])
        node_scores = []
        for node in range(num_nodes):
            # get neighbors' index
            neighbors_idx = torch.where(r_data[1]==node)[0]
            # get neighbors
            neighbors = r_data[0, neighbors_idx]
            num_neighbors = neighbors.size(0)
            neighbors_features = features[neighbors, :]
            target_features = features[node, :]
            # calculate euclid distance with broadcast
            dist: Tensor = torch.norm(neighbors_features - target_features, p=2, dim=1)
            # smaller is better and we use 'top p' in our paper 
            # => (threshold * num_neighbors)
            # see RL_neighbor_filter for details
            sorted_neighbors, sorted_index = dist.sort(descending=False)
            
            if num_neighbors <= 5: 
                remain_node_index = torch.cat((remain_node_index, neighbors_idx))
                continue # add limitations

            threshold = float(RL_thresholds[i])

            num_kept_neighbors = math.ceil(num_neighbors * threshold) + 1
            filtered_neighbors_idx = neighbors_idx[sorted_index[:num_kept_neighbors]]
            remain_node_index = torch.cat((remain_node_index, filtered_neighbors_idx))

            filtered_neighbors_scores = sorted_neighbors[:num_kept_neighbors].mean()
            node_scores.append(filtered_neighbors_scores)

        remain_node_index = remain_node_index.type('torch.LongTensor')
        edge_index = r_data[:, remain_node_index]
        multi_remain_data.append(edge_index)

        node_scores = torch.FloatTensor(node_scores) # from list
        avg_node_scores = node_scores.sum(dim=1) / num_nodes
        multi_r_score.append(avg_node_scores)

    return multi_remain_data, multi_r_score

def multi_forward_agg(args, foward_args, iter_epoch):


    # args prepare
    model, homo_data, all_num_samples, num_dim, sampler, multi_r_data, filtered_multi_r_data, device, RL_thresholds = foward_args

    if filtered_multi_r_data is None:
        filtered_multi_r_data = multi_r_data
    
    extract_features = torch.FloatTensor([])

    num_batches = int(all_num_samples / args.batch_size) + 1
    
    # all mask are then splited into mini-batch in order
    all_mask = torch.arange(0, num_dim, dtype=torch.long)

    # multiple forward with RL training
    for _ in range(iter_epoch):

        # batch training
        for batch in range(num_batches):
            start_batch = time()

            # split batch
            i_start = args.batch_size * batch
            i_end = min((batch + 1) * args.batch_size, all_num_samples)
            batch_nodes = all_mask[i_start:i_end]
            batch_labels = homo_data.y[batch_nodes]

            # sampling neighbors of batch nodes
            adjs, n_ids = sampler.sample(filtered_multi_r_data, node_idx=batch_nodes, sizes=[-1, -1], batch_size=args.batch_size)

            pred = model(homo_data.x, adjs, n_ids, device, RL_thresholds)

            extract_features = torch.cat((extract_features, pred.cpu().detach()), dim=0)

            del pred

        # RL trainig
        filtered_multi_r_data, multi_r_scores = RL_neighbor_filter_full(filtered_multi_r_data, RL_thresholds, extract_features)
        # return new RL thresholds

    return RL_thresholds

#layer
class GAT(nn.Module):
    '''
        adopt this module when using mini-batch
    '''
    def __init__(self, in_dim, hid_dim, out_dim, heads) -> None:
        super(GAT, self).__init__()
        self.GAT1 = GATConv(in_channels=in_dim, out_channels=hid_dim, heads=heads, add_self_loops=False)
        self.GAT2 = GATConv(in_channels=hid_dim * heads, out_channels=out_dim, add_self_loops=False)
        self.layers = ModuleList([self.GAT1, self.GAT2])
        self.norm = BatchNorm1d(heads * hid_dim)

        # or
        # self.GAT1 = GATConv(in_channels=in_dim, out_channels=hid_dim, heads=heads, add_self_loops=False)
        # self.GAT2 = GATConv(in_channels=hid_dim * heads, out_channels=out_dim, heads=heads, add_self_loops=False)
        # self.layers = torch.nn.ModuleList([self.GAT1, self.GAT2])
        # see Intra_AGG.forward()


    def forward(self, x, adjs, device):
        for i, (edge_index, _, size) in enumerate(adjs):
            # x: Tensor, edge_index: Tensor
            x, edge_index = x.to(device), edge_index.to(device)
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.layers[i]((x, x_target), edge_index)
            if i == 0:
                x = self.norm(x)
                x = F.elu(x)
                x = F.dropout(x, training=self.training)
            del edge_index
            
        return x

class Intra_AGG(nn.Module):

    def __init__(self, GAT_args):
        super(Intra_AGG, self).__init__()

        in_dim, hid_dim, out_dim, heads = GAT_args

        self.gnn = GAT(in_dim, hid_dim, out_dim, heads)

    def forward(self, x, adjs, device):
        
        x = self.gnn(x, adjs, device)
        
        return x

class Inter_AGG(nn.Module):

    def __init__(self, mlp_args=None):
        super(Inter_AGG, self).__init__()

        if mlp_args is not None:
            hid_dim, out_dim = mlp_args
            self.mlp = nn.Sequential(
                Linear(hid_dim, hid_dim),
                BatchNorm1d(hid_dim),
                ReLU(inplace=True),
                Dropout(),
                Linear(hid_dim, out_dim),
            )

    def forward(self, features, thresholds, inter_opt):
        
        batch_size = features[0].size(0)
        features = torch.transpose(features, dim0=0, dim1=1)
        if inter_opt == 'cat_wo_avg':
            features = features.reshape(batch_size, -1)
        elif inter_opt == 'cat_w_avg':
            # weighted average and concatenate
            features = torch.mul(features, thresholds).reshape(batch_size, -1)
        elif inter_opt == 'cat_w_avg_mlp':
            features = torch.mul(features, thresholds).reshape(batch_size, -1)
            features = self.mlp(features)
        elif inter_opt == 'cat_wo_avg_mlp':
            features = torch.mul(features, thresholds).reshape(batch_size, -1)
            features = self.mlp(features)
        elif inter_opt == 'add_wo_avg':
            features = features.sum(dim=1)
        elif inter_opt == 'add_w_avg':
            features = torch.mul(features, thresholds).sum(dim=1)
        # elif inter_opt == 'multi_avg':
            # use thresholds as the attention and remain multi-heads in last layer 
            # of GAT will improve the performance
            # features = torch.mul(features, thresholds).sum(dim=1)
        
        return features

#neighborRL
def pre_node_dist(multi_r_data, features, save_path=None):
    """This is used to culculate the similarity between node and 
    its neighbors in advance in order to avoid the repetitive computation.

    Args:
        multi_r_data ([type]): [description]
        features ([type]): [description]
        save_path ([type], optional): [description]. Defaults to None.
    """

    relation_config: Dict[str, Dict[int, Any]] = {}
    for relation_id, r_data in enumerate(multi_r_data):
        node_config: Dict[int, Any] = {}
        r_data: Tensor
        unique_nodes = r_data[1].unique()
        num_nodes = unique_nodes.size(0)
        for node in range(num_nodes):
            # get neighbors' index
            neighbors_idx = torch.where(r_data[1]==node)[0]
            # get neighbors
            neighbors = r_data[0, neighbors_idx]
            num_neighbors = neighbors.size(0)
            neighbors_features = features[neighbors, :]
            target_features = features[node, :]
            # calculate euclid distance with broadcast
            dist: Tensor = torch.norm(neighbors_features - target_features, p=2, dim=1)
            # smaller is better and we use 'top p' in our paper 
            # (threshold * num_neighbors) see RL_neighbor_filter for details
            sorted_neighbors, sorted_index = dist.sort(descending=False)
            node_config[node] = {'neighbors_idx': neighbors_idx, 
                                 'sorted_neighbors': sorted_neighbors, 
                                 'sorted_index': sorted_index, 
                                 'num_neighbors': num_neighbors}
        relation_config['relation_%d' % relation_id] = node_config

    if save_path is not None:
        save_path = os.path.join(save_path, 'relation_config.npy')
        # print(save_path)
        np.save(save_path, relation_config)

def RL_neighbor_filter(multi_r_data, RL_thresholds, load_path):
    args = args_define.args

    load_path = os.path.join(load_path, 'relation_config.npy')
    relation_config = np.load(load_path, allow_pickle=True)
    relation_config = relation_config.tolist()
    relations = list(relation_config.keys())
    multi_remain_data = []

    for i in range(len(relations)):
        print(f"Processing relation {i+1}/{len(relations)}: {relations[i]}")
        edge_index: Tensor = multi_r_data[i]
        unique_nodes = edge_index[1].unique()
        num_nodes = unique_nodes.size(0)
        remain_node_index = torch.tensor([])

        for node in range(num_nodes):
            if node % 1000 == 0:  # 每处理1000个节点输出一次进度
                print(f"  Processing node {node}/{num_nodes}")

            # extract config
            neighbors_idx = relation_config[relations[i]][node]['neighbors_idx']
            num_neighbors = relation_config[relations[i]][node]['num_neighbors']
            sorted_neighbors = relation_config[relations[i]][node]['sorted_neighbors']
            sorted_index = relation_config[relations[i]][node]['sorted_index']

            if num_neighbors <= 5: 
                remain_node_index = torch.cat((remain_node_index, neighbors_idx))
                continue # add limitations

            threshold = float(RL_thresholds[i])

            num_kept_neighbors = math.ceil(num_neighbors * threshold) + 1
            filtered_neighbors_idx = neighbors_idx[sorted_index[:num_kept_neighbors]]

            # 修正超出范围的索引
            valid_indices = filtered_neighbors_idx[filtered_neighbors_idx < edge_index.size(1)]
            remain_node_index = torch.cat((remain_node_index, valid_indices))

        remain_node_index = remain_node_index.type('torch.LongTensor')
        # print(remain_node_index)

        # Debugging print statements
        max_index = remain_node_index.max().item()
        edge_size = edge_index.size(1)
        print(f"Max remain_node_index: {max_index}")
        print(f"Edge index size: {edge_size}")

        # 修正索引超出范围的情况
        if max_index >= edge_size:
            remain_node_index = remain_node_index[remain_node_index < edge_size]

        edge_index = edge_index[:, remain_node_index]
        multi_remain_data.append(edge_index)
        print(f"Finished processing relation {relations[i]}")
    
    # 保存 multi_remain_data
    save_path = os.path.join(args.file_path, 'multi_remain_data.pt')
    torch.save(multi_remain_data, save_path)
    print(f"Filtered multi_r_data saved successfully at {save_path}")


    return multi_remain_data
    
#TripletLoss
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

class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of triplets.
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        # 确保 embeddings 至少是二维
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(1)

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        # 打印 embeddings 和 triplets 的形状用于调试
        print(f"embeddings shape: {embeddings.shape}")
        print(f"triplets shape: {triplets.shape}")

        # 检查 triplets 是否为空
        if triplets.numel() == 0:
            return torch.tensor(0.0, requires_grad=True), 0

        # 确保 triplets 的索引在 embeddings 的范围内
        if (triplets >= embeddings.size(0)).any():
            raise IndexError("triplets index out of range of embeddings")

        # 计算 anchor-positive 和 anchor-negative 的距离
        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)

def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(dim=1).view(-1, 1)
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

        #if len(triplets) == 0:
        #    triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

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


if __name__ == '__main__':
    from data_sets import Event2012_Dataset, Event2018_Dataset, MAVEN_Dataset, Arabic_Dataset

    class args_define():
        parser = argparse.ArgumentParser()
        parser.add_argument('--n_epochs', default=5, type=int, help="Number of initial-training/maintenance-training epochs.")
        parser.add_argument('--window_size', default=3, type=int, help="Maintain the model after predicting window_size blocks.")
        parser.add_argument('--patience', default=5, type=int, 
                            help="Early stop if performance did not improve in the last patience epochs.")
        parser.add_argument('--margin', default=3., type=float, help="Margin for computing triplet losses")
        parser.add_argument('--lr', default=1e-3, type=float, help="Learning rate")
        parser.add_argument('--batch_size', default=50, type=int,
                            help="Batch size (number of nodes sampled to compute triplet loss in each batch)")
        parser.add_argument('--hidden_dim', default=128, type=int, help="Hidden dimension")
        parser.add_argument('--out_dim', default=64, type=int, help="Output dimension of tweet representations")
        parser.add_argument('--heads', default=4, type=int, help="Number of heads used in GAT")
        parser.add_argument('--validation_percent', default=0.2, type=float, help="Percentage of validation nodes(tweets)")
        parser.add_argument('--use_hardest_neg', dest='use_hardest_neg', default=False, action='store_true',
                            help="If true, use hardest negative messages to form triplets. Otherwise use random ones")
        parser.add_argument('--is_shared', default=False)
        parser.add_argument('--inter_opt', default='cat_w_avg')
        parser.add_argument('--is_initial', default=True)
        parser.add_argument('--sampler', default='RL_sampler')
        parser.add_argument('--cluster_type', default='kmeans', help="Types of clustering algorithms")  # dbscan

        # RL-0
        parser.add_argument('--threshold_start0', default=[[0.2],[0.2],[0.2]], type=float,
                            help="The initial value of the filter threshold for state1 or state3")
        parser.add_argument('--RL_step0', default=0.02, type=float,
                            help="The step size of RL for state1 or state3")
        parser.add_argument('--RL_start0', default=0, type=int,
                            help="The starting epoch of RL for state1 or state3")

        # RL-1
        parser.add_argument('--eps_start', default=0.001, type=float,
                            help="The initial value of the eps for state2")
        parser.add_argument('--eps_step', default=0.02, type=float,
                            help="The step size of eps for state2")
        parser.add_argument('--min_Pts_start', default=2, type=int,
                            help="The initial value of the min_Pts for state2")
        parser.add_argument('--min_Pts_step', default=1, type=int,
                            help="The step size of min_Pts for state2")

        # other arguments
        parser.add_argument('--use_cuda', dest='use_cuda', default=True, 
                            action='store_true', help="Use cuda")
        parser.add_argument('--data_path', default='../model_saved/finevent/incremental_test/', type=str,
                            help="Path of features, labels and edges")
        parser.add_argument('--file_path', default='../model_saved/finevent/', type=str,
                            help="Path of files to save")
        # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
        parser.add_argument('--mask_path', default=None, type=str,
                            help="File path that contains the training, validation and test masks")
        # format: './incremental_0808/incremental_graphs_0808/embeddings_XXXX'
        parser.add_argument('--log_interval', default=10, type=int,
                            help="Log interval")
        args = parser.parse_args()

    args = args_define.args
    dataset = Event2012_Dataset.load_data()
    finevent = FinEvent(args,dataset)

    #finevent.preprocess()
    finevent.fit()
    predictions, ground_truths = finevent.detection()

    # Evaluate model
    finevent.evaluate(predictions, ground_truths)

 




