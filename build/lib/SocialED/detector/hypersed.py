import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re 
import torch
import random
import json
import numpy as np
import math
import pickle
from datetime import datetime
from collections import Counter, namedtuple
from queue import Queue
from typing import List
from random import sample
from scipy import sparse
from sentence_transformers import SentenceTransformer
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.cluster import KMeans
from munkres import Munkres
#from math import log2, sqrt
from tqdm import tqdm
from torch_geometric.utils import negative_sampling, dropout_edge, add_self_loops, from_networkx
from torch_scatter import scatter_sum, scatter_softmax
import pandas as pd
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from itertools import combinations

import geoopt
import geoopt.manifolds.lorentz.math as lmath
from geoopt.manifolds import PoincareBall
from geoopt.manifolds.stereographic.math import mobius_matvec, project, expmap0, mobius_add, logmap0, dist0, dist
from geoopt.tensor import ManifoldParameter
from geoopt.optim import RiemannianAdam
from os.path import exists
from networkx.algorithms import cuts
from itertools import chain



# Set random seeds
seed = 3047
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)



class HyperSED:
    r"""The HyperSED model for social event detection that uses hyperbolic graph neural networks
    and deep structure inference for event detection.

    .. note::
        This detector uses hyperbolic graph neural networks and deep structure inference to identify events in social media data.
        The model requires a dataset object with a load_data() method.

    Parameters
    ----------
    dataset : object
        The dataset object containing social media data.
        Must provide load_data() method that returns the raw data.
    devices : bool, optional
        Whether to use GPU if available. Default: ``True``.
    algorithm : str, optional
        Algorithm name. Default: ``"HyperSED"``.
    data_path : str, optional
        Path to preprocessed data. Default: ``'../model/model_saved/hypersed/datasets/data_preprocess'``.
    save_model_path : str, optional
        Path to save model files. Default: ``'../model/model_saved/hypersed/datasets/data_preprocess/saved_models'``.
    n_cluster_trials : int, optional
        Number of clustering trials. Default: ``5``.
    encode : str, optional
        Text encoding method. Default: ``'SBERT'``.
    edge_type : str, optional
        Type of edges to use. Default: ``'e_as'``.
    gpu : int, optional
        GPU device ID to use. Default: ``0``.
    num_epochs : int, optional
        Number of training epochs. Default: ``50``.
    patience : int, optional
        Early stopping patience. Default: ``50``.
    hgae : bool, optional
        Whether to use hyperbolic GAE. Default: ``True``.
    dsi : bool, optional
        Whether to use deep structure inference. Default: ``True``.
    pre_anchor : bool, optional
        Whether to use pre-anchoring. Default: ``True``.
    anchor_rate : int, optional
        Anchor sampling rate. Default: ``20``.
    plot : bool, optional
        Whether to plot results. Default: ``False``.
    thres : float, optional
        Threshold value. Default: ``0.5``.
    diag : float, optional
        Diagonal value. Default: ``0.5``.
    num_layers_gae : int, optional
        Number of GAE layers. Default: ``2``.
    hidden_dim_gae : int, optional
        Hidden dimension for GAE. Default: ``128``.
    out_dim_gae : int, optional
        Output dimension for GAE. Default: ``2``.
    t : float, optional
        Temperature parameter. Default: ``1.0``.
    r : float, optional
        Curvature radius. Default: ``2.0``.
    lr_gae : float, optional
        Learning rate for GAE. Default: ``1e-3``.
    w_decay : float, optional
        Weight decay. Default: ``0.3``.
    dropout : float, optional
        Dropout rate. Default: ``0.4``.
    nonlin : optional
        Non-linear activation function. Default: ``None``.
    use_attn : bool, optional
        Whether to use attention. Default: ``False``.
    use_bias : bool, optional
        Whether to use bias. Default: ``True``.
    decay_rate : optional
        Learning rate decay rate. Default: ``None``.
    num_layers : int, optional
        Number of layers. Default: ``3``.
    hidden_dim : int, optional
        Hidden dimension. Default: ``64``.
    out_dim : int, optional
        Output dimension. Default: ``2``.
    height : int, optional
        Tree height. Default: ``2``.
    max_nums : list, optional
        Maximum numbers per level. Default: ``[300]``.
    temperature : float, optional
        Temperature for loss. Default: ``0.05``.
    lr_pre : float, optional
        Learning rate for pre-training. Default: ``1e-3``.
    lr : float, optional
        Learning rate. Default: ``1e-3``.
    """
    
    def __init__(self,
                 dataset,
                 devices=True,
                 algorithm="HyperSED",
                 data_path='../model/model_saved/hypersed/datasets/data_preprocess',
                 save_model_path='../model/model_saved/hypersed/datasets/data_preprocess/saved_models',
                 n_cluster_trials=5,

                 encode='SBERT',
                 edge_type='e_as',
                 gpu=0,
                 num_epochs=50,
                 patience=50,
                 hgae=True,
                 dsi=True,
                 pre_anchor=True,
                 anchor_rate=20,
                 plot=False,
                 thres=0.5,
                 diag=0.5,
                 # HGAE params 
                 num_layers_gae=2,
                 hidden_dim_gae=128,
                 out_dim_gae=2,
                 t=1.,
                 r=2.,
                 lr_gae=1e-3,
                 w_decay=0.3,
                 dropout=0.4,
                 nonlin=None,
                 use_attn=False,
                 use_bias=True,
                 # DSI params
                 decay_rate=None,
                 num_layers=3,
                 hidden_dim=64,
                 out_dim=2,
                 height=2,
                 max_nums=[300],
                 temperature=0.05,
                 lr_pre=1e-3,
                 lr=1e-3):

        self.dataset = dataset 
        self.dataset_name = dataset.get_dataset_name()
        self.devices = devices
        self.algorithm = algorithm
        self.data_path = data_path
        self.save_model_path = save_model_path
        self.n_cluster_trials = n_cluster_trials


        self.encode = encode
        self.edge_type = edge_type
        self.gpu = gpu
        self.num_epochs = num_epochs
        self.patience = patience
        self.hgae = hgae
        self.dsi = dsi
        self.pre_anchor = pre_anchor
        self.anchor_rate = anchor_rate
        self.plot = plot
        self.thres = thres
        self.diag = diag

        # HGAE params
        self.num_layers_gae = num_layers_gae
        self.hidden_dim_gae = hidden_dim_gae
        self.out_dim_gae = out_dim_gae
        self.t = t
        self.r = r
        self.lr_gae = lr_gae
        self.w_decay = w_decay
        self.dropout = dropout
        self.nonlin = nonlin
        self.use_attn = use_attn
        self.use_bias = use_bias

        # DSI params
        self.decay_rate = decay_rate
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.height = height
        self.max_nums = max_nums
        self.temperature = temperature
        self.lr_pre = lr_pre
        self.lr = lr

        # Set algorithm name
        self.algorithm_name = '_'.join([name for name, value in vars(self).items() 
                                    if value and name in ['hgae', 'pre_anchor', 'dsi']])


    def preprocess(self):
        preprocessor = Preprocessor(self.dataset)
        preprocessor.preprocess()


    def fit(self):
        # 检查 GPU 可用性并设置适当的设备
        if self.devices and torch.cuda.is_available():
            if self.gpu >= torch.cuda.device_count():
                print(f"Warning: GPU {self.gpu} not available, falling back to CPU")
                self.devices = False
                self.gpu = -1
            else:
                torch.cuda.set_device(self.gpu)
                print(f"Using GPU: {torch.cuda.get_device_name(self.gpu)}")
        else:
            self.devices = False
            self.gpu = -1
            print("Using CPU")

        self.trainer = Trainer(self)
        self.trainer.fit()

        
    def detection(self):
        ground_truths, predictions = self.trainer.detection()
        torch.cuda.empty_cache()
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


class Trainer(nn.Module):
    def __init__(self, args, block=None):
        super(Trainer, self).__init__()

        self.args = args
        self.block = block
        self.data = TwitterDataSet(args, args.dataset_name, block)
        self.device = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.devices else "cpu"
        self.save_model_path = f"{args.save_model_path}/{args.algorithm}/{args.dataset_name}/best_model"
        self.manifold = Lorentz()

        self.in_dim = self.data.data.feature.shape[1]
        self.num_nodes = self.data.data.num_nodes
        
        if self.args.hgae:
            # Part 1  Hyper Graph Encoder
            self.gae = HyperGraphAutoEncoder(self.args, self.device, self.manifold, args.num_layers_gae, 
                                            self.in_dim, args.hidden_dim_gae, args.out_dim_gae, args.dropout, 
                                            args.nonlin, args.use_attn, args.use_bias).to(self.device)
            if self.args.dsi:
                self.in_dim = args.out_dim_gae

        if self.args.dsi:
            # Part 2  Hyper Structure Entropy
            self.hyperSE = HyperSE(args=self.args, manifold=self.manifold, n_layers=args.num_layers, device=self.device, 
                                   in_features=self.in_dim, hidden_dim_enc=args.hidden_dim, hidden_features=args.hidden_dim, 
                                   num_nodes=self.num_nodes, height=args.height, temperature=args.temperature, embed_dim=args.out_dim,
                                   dropout=args.dropout, nonlin=args.nonlin, decay_rate=args.decay_rate, 
                                   max_nums=args.max_nums, use_att=args.use_attn, use_bias=args.use_bias).to(self.device)

        self.patience = self.args.patience


    def forward(self, data, mode="val"):
        # for testing, with no loss
        with torch.no_grad():
            if self.args.hgae:
                loss, feature = self.getGAEPre(data, mode)
            else:
                feature = data.anchor_feature if self.args.pre_anchor else data.feature
            adj = data.anchor_edge_index_types.adj if self.args.pre_anchor else data.edge_index_types.adj
            if self.args.dsi:
                feature = self.hyperSE(feature, adj)

        return feature.detach().cpu()


    def fit(self):
        """Train the model"""
        self.train()
        time1 = datetime.now().strftime("%H:%M:%S")
        epochs = self.args.num_epochs
        data = self.data.data
        self.best_cluster = {'block_id': self.block, 'nmi': 0, 'ami': 0, 'ari': 0}

        # training the new block
        for epoch in tqdm(range(epochs), desc="Training Epochs"):
            if self.args.hgae:
                self.gae.optimizer.zero_grad()
            if self.args.dsi:
                self.hyperSE.optimizer_pre.zero_grad()
                if epoch > 0:
                    self.hyperSE.optimizer.zero_grad()

            # Part 1  Hyper Graph AutoEncoder
            if self.args.hgae:
                loss, feature = self.getGAEPre(data)
            else:
                feature = None

            # Part 2  Hyper Structural Entropy
            if self.args.dsi:
                input_data = self.getOtherByedge(data, feature, epoch)
                hse_loss = self.hyperSE.loss(input_data)
                if self.args.hgae:
                    loss = loss + hse_loss
                else:
                    loss = hse_loss
            
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            loss.backward()
            if self.args.hgae:
                self.gae.optimizer.step()
            if self.args.dsi:
                self.hyperSE.optimizer_pre.step()
                if epoch > 0:
                    self.hyperSE.optimizer.step()

            # Evaluate current model
            self.eval()
            with torch.no_grad():
                embeddings = self.forward(data, "val")

                if self.args.dsi:
                    manifold = self.hyperSE.manifold.cpu()
                    tree = construct_tree(torch.tensor([i for i in range(embeddings.shape[0])]).long(),
                                        manifold,
                                        self.hyperSE.embeddings, self.hyperSE.ass_mat, height=self.args.height,
                                        num_nodes=embeddings.shape[0])
                    tree_graph = to_networkx_tree(tree, manifold, height=self.args.height)
                    predicts = decoding_cluster_from_tree(manifold, tree_graph,
                                                        data.num_classes, embeddings.shape[0],
                                                        height=self.args.height)
                else:
                    fea = self.manifold.to_poincare(embeddings)
                    Z_np = fea.detach().cpu().numpy()
                    M = data.num_classes

                    kmeans = PoincareKMeans(n_clusters=M, n_init=1, max_iter=200, tol=1e-10, verbose=True)
                    kmeans.fit(Z_np)
                    predicts = kmeans.labels_

                trues = data.labels
                nmis, amis, aris = [], [], []
                if self.args.pre_anchor:
                    predicts = getNewPredict(predicts, data.anchor_ass)
                for step in range(self.args.n_cluster_trials):
                    metrics = cluster_metrics(trues, predicts.astype(int))
                    nmi, ami, ari = metrics.evaluateFromLabel()
                    nmis.append(nmi)
                    amis.append(ami)
                    aris.append(ari)

                nmi, ami, ari = np.mean(nmis), np.mean(amis), np.mean(aris)
                
                if nmi >= self.best_cluster['nmi'] and ami >= self.best_cluster['ami'] and ari >= self.best_cluster['ari']:
                    self.patience = self.args.patience

                    model_path = f'{self.save_model_path}/{self.block}'
                    if not os.path.exists(model_path):
                        os.makedirs(model_path)
                    torch.save(self.state_dict(), f"{model_path}/model.pt")
                    
                    self.best_cluster['nmi'] = nmi
                    self.best_cluster['ami'] = ami
                    self.best_cluster['ari'] = ari
                    self.best_cluster['predicts'] = predicts
                    self.best_cluster['trues'] = trues

                elif nmi < self.best_cluster['nmi'] and ami < self.best_cluster['ami'] and ari < self.best_cluster['ari']:
                    self.patience -= 1

                else:
                    if nmi > self.best_cluster['nmi']:
                        print(f'nmi: {nmi}, ami: {ami}, ari: {ari}')
                    elif ami > self.best_cluster['ami']:
                        print(f'ami: {ami}, nmi: {nmi}, ari: {ari}')
                    elif ari > self.best_cluster['ari']:
                        print(f'ari: {ari}, nmi: {nmi}, ami: {ami}')
                    else:
                        print()

            print(f"VALID {self.block} : NUM_MSG: {data.num_nodes}, NMI: {self.best_cluster['nmi']}, AMI: {self.best_cluster['ami']}, ARI: {self.best_cluster['ari']}")

            torch.cuda.empty_cache() 

            if self.patience < 0:
                print("Run out of patience")
                break

            self.train()

        time2 = datetime.now().strftime("%H:%M:%S")
        self.time = {'t3': time1, 't4': time2}

    def detection(self):
        """Detect communities and return results"""
        return self.best_cluster['trues'], self.best_cluster['predicts']


    def getGAEPre(self, data, mode="train"):
        # get hgae latent representation
        if self.args.pre_anchor:
            features = data.anchor_feature.clone()
            adj_ori = data.anchor_edge_index_types.adj.clone()
        else:
            features = data.feature.clone()
            adj_ori = data.edge_index_types.adj.clone()
        
        if mode == "train":
            loss, adj, feature = self.gae.loss(features, adj_ori)
        else:
            adj, feature = self.gae.forward(features, adj_ori)
            loss = None
        
        return loss, feature
    

    def getOtherByedge(self, data, gae_feature, epoch):
        if self.args.hgae:
            feature = gae_feature
            if self.args.pre_anchor:
                edge_index_types = data.anchor_edge_index_types
            else:
                edge_index_types = data.edge_index_types
        else:
            if self.args.pre_anchor:
                feature = data.anchor_feature
                edge_index_types = data.anchor_edge_index_types
            else:
                feature = data.feature
                edge_index_types = data.edge_index_types
        return DSIData(feature=feature, adj=edge_index_types.adj, weight=edge_index_types.weight, degrees=edge_index_types.degrees, 
                        neg_edge_index=edge_index_types.neg_edge_index, edge_index=edge_index_types.edge_index, device=self.device, 
                        pretrain=True if epoch == 0 else False)

    # 离线模式学习
    def offline_train(self):
        pass

    # 开始测试
    def test(self, data, block_i):
        model_path = f'{self.save_model_path}/{block_i}'
        self.load_state_dict(torch.load(model_path + "/model.pt"))
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(data, "test")
            manifold = self.hyperSE.manifold.cpu()

            if self.args.dsi:
                tree = construct_tree(torch.tensor([i for i in range(embeddings.shape[0])]).long(),
                                    manifold,
                                    self.hyperSE.embeddings, self.hyperSE.ass_mat, height=self.args.height,
                                    num_nodes=embeddings.shape[0])
                tree_graph = to_networkx_tree(tree, manifold, height=self.args.height)
                if self.args.plot:
                    labels = data.anchor_labels if self.args.pre_anchor else data.labels
                    _, color_dict = plot_leaves(tree_graph, manifold, embeddings, labels, height=self.args.height,
                                                save_path='/home/yuxiaoyan/paper/HyperSED/' + f"{self.args.height}_{1}_true.pdf")

                predicts = decoding_cluster_from_tree(manifold, tree_graph,
                                                    data.num_classes, embeddings.shape[0],
                                                    height=self.args.height)
            
            else:
                fea = self.manifold.to_poincare(embeddings)
                Z_np = fea.detach().cpu().numpy()
                M = data.num_classes

                kmeans = PoincareKMeans(n_clusters=M, n_init=1, max_iter=200, tol=1e-10, verbose=True)
                kmeans.fit(Z_np)
                predicts = kmeans.labels_
            

            trues = data.labels

            nmis, amis, aris = [], [], []
            if self.args.pre_anchor:
                predicts = getNewPredict(predicts, data.anchor_ass)
            for step in range(self.args.n_cluster_trials):
                metrics = cluster_metrics(trues, predicts)
                nmi, ami, ari = metrics.evaluateFromLabel()
                nmis.append(nmi)
                amis.append(ami)
                aris.append(ari)

            nmi, ami, ari = np.mean(nmis), np.mean(amis), np.mean(aris)
            metrics.get_new_predicts()
            new_pred = metrics.new_predicts
            plot_leaves(tree_graph, manifold, embeddings, new_pred, height=self.args.height,
                        save_path='../model/model_saved/hypersed/' + f"{self.args.height}_{1}_pred.pdf",
                        colors_dict=color_dict)

        return nmi, ami, ari


    # 获取训练集以及验证集数据
    def getNTBlockData(self, datas, block_id):
        # views = list(Views._fields)
        data = datas[block_id]
        features, labels, num_classes, num_features, num_nodes = data.feature, data.labels, data.num_classes, data.num_features, data.num_nodes

        datasets = {}
        # for view in views:
        edge_index_type = data.edge_index_types
        # 划分训练集、测试集以及验证集
        pos_edges, neg_edges = mask_edges(edge_index_type.edge_index, edge_index_type.neg_edge_index, 0.1, 0.2 if block_id == 0 else 0)

        train_adj, train_degrees, train_weight = getOtherByedge(pos_edges[0], num_nodes)
        val_adj, val_train_degrees, val_train_weight = getOtherByedge(pos_edges[1], num_nodes)
        train_edge = EdgeIndexTypes(adj=train_adj, degrees=train_degrees, weight=train_weight, edge_index=pos_edges[0], neg_edge_index=neg_edges[0])
        val_edge = EdgeIndexTypes(adj=val_adj, degrees=val_train_degrees, weight=val_train_weight, edge_index=pos_edges[1], neg_edge_index=neg_edges[1])
        if block_id == 0:
            test_adj, test_train_degrees, test_train_weight = getOtherByedge(pos_edges[1], num_nodes)
            test_edge = EdgeIndexTypes(adj=test_adj, degrees=test_train_degrees, weight=test_train_weight, edge_index=pos_edges[2], neg_edge_index=neg_edges[2])
            datasets = {"train": train_edge, "val":  val_edge, "test":  test_edge}
        else:
            test_edge_index_type = EdgeIndexTypes(adj=edge_index_type.adj, degrees=edge_index_type.degrees, weight=edge_index_type.weight, edge_index=edge_index_type.edge_index, neg_edge_index=edge_index_type.neg_edge_index)
            datasets = {"train": train_edge, "val":  val_edge, "test": test_edge_index_type}

        dealed_data = DataPartition(features=features, labels=labels, num_nodes=num_nodes, num_classes=num_classes, num_features=num_features, views=datasets)
        return dealed_data


    def getOtherByedge(self, data, gae_feature, epoch):
        if self.args.hgae:
            feature = gae_feature
            if self.args.pre_anchor:
                edge_index_types = data.anchor_edge_index_types
            else:
                edge_index_types = data.edge_index_types
        else:
            if self.args.pre_anchor:
                feature = data.anchor_feature
                edge_index_types = data.anchor_edge_index_types
            else:
                feature = data.feature
                edge_index_types = data.edge_index_types
        return DSIData(feature=feature, adj=edge_index_types.adj, weight=edge_index_types.weight, degrees=edge_index_types.degrees, 
                        neg_edge_index=edge_index_types.neg_edge_index, edge_index=edge_index_types.edge_index, device=self.device, 
                        pretrain=True if epoch == 0 else False)
    

class Preprocessor:
    def __init__(self, dataset):
        """Initialize preprocessor
        Args:
            dataset: Dataset calss (e.g. Event2012, Event2018, etc.)
            language: Language of the dataset (default 'English')
        """
        self.dataset = dataset.load_data()
        self.language = dataset.get_dataset_language()
        self.dataset_name = dataset.get_dataset_name()
        self.columns = ['tweet_id', 'text', 'event_id', 'words', 'filtered_words',
                       'entities', 'user_id', 'created_at', 'urls', 'hashtags', 'user_mentions']

    def get_set_test_df(self, df):
        """Get closed set test dataframe"""
        save_path = f'../model/model_saved/hypersed/{self.dataset_name}/data/'
        if not exists(save_path):
            os.makedirs(save_path)
        
        test_set_df_np_path = save_path + 'test_set.npy'
        test_set_label_path = save_path + 'label.npy'

        if not exists(test_set_df_np_path):
            # Use 2012-style processing for all datasets
            test_mask = torch.load(f'../model/model_saved/hypersed/{self.dataset_name}/masks/test_mask.pt').cpu().detach().numpy()
            test_mask = list(np.where(test_mask==True)[0])
            test_df = df.iloc[test_mask]
            shuffled_index = np.random.permutation(test_df.index)
            shuffled_df = test_df.reindex(shuffled_index)
            shuffled_df.reset_index(drop=True, inplace=True)
            
            test_df_np = test_df.to_numpy()
            labels = [int(label) for label in shuffled_df['event_id'].values]
            
            np.save(test_set_label_path, np.asarray(labels))
            np.save(test_set_df_np_path, test_df_np)
        

    def get_set_messages_embeddings(self):
        """Get SBERT embeddings for closed set messages"""
        save_path = f'../model/model_saved/hypersed/{self.dataset_name}/data/'
        
        SBERT_embedding_path = f'{save_path}/SBERT_embeddings.pkl'
        if not exists(SBERT_embedding_path):
            test_set_df_np_path = save_path + 'test_set.npy'
            test_df_np = np.load(test_set_df_np_path, allow_pickle=True)
            
            test_df = pd.DataFrame(data=test_df_np, columns=self.columns)
            print("Dataframe loaded.")

            processed_text = [preprocess_sentence(s) for s in test_df['text'].values]
            print('message text contents preprocessed.')

            embeddings = SBERT_embed(processed_text, language=self.language)

            with open(SBERT_embedding_path, 'wb') as fp:
                pickle.dump(embeddings, fp)
            print('SBERT embeddings stored.')
        return


    def preprocess(self):
        """Main preprocessing function"""
        # Load raw data using 2012-style processing
        df_np = self.dataset
        
        print("Loaded data.")
        df = pd.DataFrame(data=df_np, columns=self.columns)
        print("Data converted to dataframe.")

        save_dir = os.path.join(f'../model/model_saved/hypersed/{self.dataset_name}', 'masks')
        os.makedirs(save_dir, exist_ok=True)
        
        # Split and save masks
        self.split_and_save_masks(df, save_dir)
        print("Generated and saved train/val/test masks.")

        self.get_set_test_df(df)
        self.get_set_messages_embeddings()
        self.construct_graph_all(self.dataset_name)

        return

    def split_and_save_masks(self, df, save_dir, train_size=0.7, val_size=0.1, test_size=0.2, random_seed=42):
        """
        Splits the DataFrame into training, validation, and test sets, and saves the indices (masks) as .pt files.
        
        Parameters:
        - df (pd.DataFrame): The DataFrame to be split
        - save_dir (str): Directory to save the masks
        - train_size (float): Proportion for training (default 0.7)
        - val_size (float): Proportion for validation (default 0.1) 
        - test_size (float): Proportion for testing (default 0.2)
        - random_seed (int): Random seed for reproducibility
        """
        if train_size + val_size + test_size != 1.0:
            raise ValueError("train_size + val_size + test_size must equal 1.0")

        if df.empty:
            raise ValueError("The input DataFrame is empty.")

        print(f"Total samples in DataFrame: {len(df)}")
        
        # Set random seed
        torch.manual_seed(random_seed)

        # Split into train and temp
        train_data, temp_data = train_test_split(df, train_size=train_size, random_state=random_seed)
        
        # Split temp into val and test
        val_data, test_data = train_test_split(temp_data, 
                                             train_size=val_size/(val_size + test_size),
                                             random_state=random_seed)

        # Create boolean masks
        full_train_mask = torch.zeros(len(df), dtype=torch.bool)
        full_val_mask = torch.zeros(len(df), dtype=torch.bool)
        full_test_mask = torch.zeros(len(df), dtype=torch.bool)

        # Set indices
        full_train_mask[train_data.index] = True
        full_val_mask[val_data.index] = True  
        full_test_mask[test_data.index] = True

        print(f"Training samples: {full_train_mask.sum()}")
        print(f"Validation samples: {full_val_mask.sum()}")
        print(f"Test samples: {full_test_mask.sum()}")

        # Save masks
        mask_paths = {
            'train_mask.pt': full_train_mask,
            'val_mask.pt': full_val_mask, 
            'test_mask.pt': full_test_mask
        }

        for filename, mask in mask_paths.items():
            mask_path = os.path.join(save_dir, filename)
            if not os.path.exists(mask_path):
                try:
                    torch.save(mask, mask_path)
                    print(f"Saved {filename}")
                except Exception as e:
                    print(f"Error saving {filename}: {e}")
            else:
                print(f"{filename} already exists")

            # Verify saved file
            if os.path.exists(mask_path):
                saved_mask = torch.load(mask_path)
                if saved_mask.numel() == 0:
                    print(f"Warning: {filename} is empty")
                else:
                    print(f"Verified {filename} with {saved_mask.numel()} elements")

        print("Mask generation completed")

    # Construct message graph
    def get_best_threshold(self, path):
        best_threshold_path = path + '/best_threshold.pkl'
        if not os.path.exists(best_threshold_path):
            embeddings_path = path + '/SBERT_embeddings.pkl'
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            best_threshold = search_threshold(embeddings)
            best_threshold = {'best_thres': best_threshold}
            with open(best_threshold_path, 'wb') as fp:
                pickle.dump(best_threshold, fp)
            print('best threshold is stored.')

        with open(best_threshold_path, 'rb') as f:
            best_threshold = pickle.load(f)
        print('best threshold loaded.')
        return best_threshold


    def construct_graph(self, df, embeddings, save_path, e_a=True, e_s=True):
        """Construct graph for given dataframe and embeddings"""
        def safe_list(x):
            """Convert input to list safely"""
            if isinstance(x, (list, tuple, set)):
                return x
            elif x is None:
                return []
            else:
                return [x]

        def safe_str_lower(x):
            """Convert input to lowercase string safely"""
            try:
                return str(x).lower()
            except:
                return str(x)

        # Use unified columns
        all_node_features = [
            [str(u)] + 
            [str(each) for each in safe_list(um)] + 
            [safe_str_lower(h) for h in safe_list(hs)] + 
            (e if isinstance(e, list) else [str(e)])
            for u, um, hs, e in zip(df['user_id'], df['user_mentions'], 
                                   df['hashtags'], df['entities'])
        ]
        
        best_threshold = self.get_best_threshold(save_path)
        best_threshold = best_threshold['best_thres']

        global_edges = get_global_edges(all_node_features, embeddings, best_threshold, e_a=e_a, e_s=e_s)

        corr_matrix = np.corrcoef(embeddings)
        np.fill_diagonal(corr_matrix, 0)
        weighted_global_edges = [(edge[0], edge[1], corr_matrix[edge[0], edge[1]]) 
                                for edge in global_edges if corr_matrix[edge[0], edge[1]] > 0]
        
        edge_types = 'e_as' if e_s and e_a else 'e_s' if e_s else 'e_a' if e_a else None

        # Create adjacency matrix
        num_nodes = embeddings.shape[0]
        adj_matrix = np.zeros((num_nodes, num_nodes))
        for node1, node2, weight in weighted_global_edges:
            adj_matrix[node1, node2] = weight
            adj_matrix[node2, node1] = weight
            
        sparse_adj_matrix = sparse.csr_matrix(adj_matrix)
        return sparse_adj_matrix, edge_types

    def construct_graph_all(self, dataset_name, e_a=True, e_s=True):
        save_path = f'../model/model_saved/hypersed/{self.dataset_name}/data/'

        # Unified columns
        columns = ['tweet_id', 'text', 'event_id', 'words', 'filtered_words',
                'entities', 'user_id', 'created_at', 'urls', 'hashtags', 'user_mentions']

        print('\n\n====================================================')
        time1 = datetime.now().strftime("%H:%M:%S")
        print('Graph construct starting time:', time1)

        # Load embeddings and data
        with open(f'{save_path}/SBERT_embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        
        df_np = np.load(f'{save_path}/test_set.npy', allow_pickle=True)
        df = pd.DataFrame(data=df_np, columns=columns)

        # Construct graph
        sparse_adj_matrix, edge_types = self.construct_graph(df, embeddings, save_path, e_a, e_s)
        sparse.save_npz(f'{save_path}/message_graph_{edge_types}.npz', sparse_adj_matrix)

        time2 = datetime.now().strftime("%H:%M:%S")
        print('Graph construct ending time:', time2)



def mobius_add(x, y):
    """Mobius addition in numpy."""
    xy = np.sum(x * y, 1, keepdims=True)
    x2 = np.sum(x * x, 1, keepdims=True)
    y2 = np.sum(y * y, 1, keepdims=True)
    num = (1 + 2 * xy + y2) * x + (1 - x2) * y
    den = 1 + 2 * xy + x2 * y2
    return num / den


def mobius_mul(x, t):
    """Mobius multiplication in numpy."""
    normx = np.sqrt(np.sum(x * x, 1, keepdims=True))
    return np.tanh(t * np.arctanh(normx)) * x / normx


def geodesic_fn(x, y, nb_points=100):
    """Get coordinates of points on the geodesic between x and y."""
    t = np.linspace(0, 1, nb_points)
    x_rep = np.repeat(x.reshape((1, -1)), len(t), 0)
    y_rep = np.repeat(y.reshape((1, -1)), len(t), 0)
    t1 = mobius_add(-x_rep, y_rep)
    t2 = mobius_mul(t1, t.reshape((-1, 1)))
    return mobius_add(x_rep, t2)


def plot_geodesic(x, y, ax):
    """Plots geodesic between x and y."""
    points = geodesic_fn(x, y)
    ax.plot(points[:, 0], points[:, 1], color='black', linewidth=0.3, alpha=1.)


def plot_leaves(tree, manifold, embeddings, labels, height, save_path=None, colors_dict=None):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    circle = plt.Circle((0, 0), 1.0, color='y', alpha=0.1)
    ax.add_artist(circle)
    for k in range(1, height + 1):
        circle_k = plt.Circle((0, 0), k / (height + 1), color='b', alpha=0.05)
        ax.add_artist(circle_k)
    n = embeddings.shape[0]
    colors_dict = get_colors(labels, color_seed=1234) if colors_dict is None else colors_dict
    colors = [colors_dict[k] for k in labels]
    embeddings = manifold.to_poincare(embeddings).numpy()
    scatter = ax.scatter(embeddings[:n, 0], embeddings[:n, 1], c=colors, s=80, alpha=1.0)
    # legend = ax.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    # ax.add_artist(legend)
    # ax.scatter(np.array([0]), np.array([0]), c='black')
    for u, v in tree.edges():
        x = manifold.to_poincare(tree.nodes[u]['coords']).numpy()
        y = manifold.to_poincare(tree.nodes[v]['coords']).numpy()
        if tree.nodes[u]['is_leaf'] is False:
            c = 'black' if tree.nodes[u]['height'] == 0 else 'red'
            m = '*' if tree.nodes[u]['height'] == 0 else 's'
            ax.scatter(x[0], x[1], c=c, s=30, marker=m)
        if tree.nodes[v]['is_leaf'] is False:
            c = 'black' if tree.nodes[v]['height'] == 0 else 'red'
            m = '*' if tree.nodes[u]['height'] == 0 else 's'
            ax.scatter(y[0], y[1], c=c, s=30, marker=m)
        plot_geodesic(y, x, ax)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis("off")
    plt.savefig(save_path, transparent=True, bbox_inches='tight', dpi=500)
    plt.show()
    return ax, colors_dict


def get_colors(y, color_seed=1234):
    """random color assignment for label classes."""
    np.random.seed(color_seed)
    colors = {}
    for k in np.unique(y):
        r = np.random.random()
        b = np.random.random()
        g = np.random.random()
        colors[k] = (r, g, b)
    return colors


def plot_nx_graph(G: nx.Graph, root, save_path=None):
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(111)
    pos = graphviz_layout(G, 'twopi')
    nx.draw(G, pos, ax=ax, with_labels=True)
    plt.savefig(save_path)
    plt.show()



Dataset = namedtuple('Dataset', ['data', 'embedding', "path"])

EdgeIndexTypes = namedtuple('EdgeIndexTypes', ['edge_index', 'weight', 'degrees', 
                                               'neg_edge_index', 'adj'])

SingleBlockData = namedtuple('SingleBlockData', ['feature', 'num_features', 'labels', 
                                                 'num_nodes', 'num_classes', 
                                                 'edge_index_types', 'anchor_feature', 'num_anchors', 
                                                 'anchor_edge_index_types', 'anchor_ass', 'anchor_labels'])

DSIData = namedtuple('DSIData', ['edge_index', 'device', 'pretrain', 'weight', 'adj', 
                                   'feature', 'degrees', 'neg_edge_index'])

DataPartition = namedtuple('DataPartition', ['features', 'labels', 'num_classes', 
                                             'num_features', 'views', 'num_nodes'])

Views = namedtuple('Views', ['userid', 'word', 'entity', 'all'])

SingleBlockData.__annotations__ = {
    'feature': np.ndarray,  # 假设feature是一个NumPy数组
    'num_features': int,
    'labels': np.ndarray,   # 假设labels也是一个NumPy数组
    'num_nodes': int,
    'num_classes': int,
    'edge_index_types': EdgeIndexTypes,  # 假设edge_index_types是一个字符串列表
    'adj': None
}



def decoding_cluster_from_tree(manifold, tree: nx.Graph, num_clusters, num_nodes, height):
    root = tree.nodes[num_nodes]
    root_coords = root['coords']
    dist_dict = {}  # for every height of tree
    for u in tree.nodes():
        if u != num_nodes:  # u is not root
            h = tree.nodes[u]['height']
            dist_dict[h] = dist_dict.get(h, {})
            dist_dict[h].update({u: manifold.dist(root_coords, tree.nodes[u]['coords']).numpy()})

    h = 1
    sorted_dist_list = sorted(dist_dict[h].items(), reverse=False, key=lambda x: x[1])
    count = len(sorted_dist_list)
    group_list = [([u], dist) for u, dist in sorted_dist_list]  # [ ([u], dist_u) ]
    while len(group_list) <= 1:
        h = h + 1
        sorted_dist_list = sorted(dist_dict[h].items(), reverse=False, key=lambda x: x[1])
        count = len(sorted_dist_list)
        group_list = [([u], dist) for u, dist in sorted_dist_list]

    while count > num_clusters:
        group_list, count = merge_nodes_once(manifold, root_coords, tree, group_list, count)

    while count < num_clusters and h <= height:
        h = h + 1   # search next level
        pos = 0
        while pos < len(group_list):
            v1, d1 = group_list[pos]  # node to split
            sub_level_set = []
            v1_coord = tree.nodes[v1[0]]['coords']
            for u, v in tree.edges(v1[0]):
                if tree.nodes[v]['height'] == h:
                    v_coords = tree.nodes[v]['coords']
                    dist = manifold.dist(v_coords, v1_coord).cpu().numpy()
                    sub_level_set.append(([v], dist))    # [ ([v], dist_v) ]
            if len(sub_level_set) <= 1:
                pos += 1
                continue
            sub_level_set = sorted(sub_level_set, reverse=False, key=lambda x: x[1])
            count += len(sub_level_set) - 1
            if count > num_clusters:
                while count > num_clusters:
                    sub_level_set, count = merge_nodes_once(manifold, v1_coord, tree, sub_level_set, count)
                del group_list[pos]  # del the position node which will be split
                group_list += sub_level_set    # Now count == num_clusters
                break
            elif count == num_clusters:
                del group_list[pos]  # del the position node which will be split
                group_list += sub_level_set
                break
            else:
                del group_list[pos]
                group_list += sub_level_set
                pos += 1

    cluster_dist = {}
    # for i in range(num_clusters):
    #     u_list, _ = group_list[i]
    #     group = []
    #     for u in u_list:
    #         index = tree.nodes[u]['children'].tolist()
    #         group += index
    #     cluster_dist.update({k: i for k in group})
    for i in range(len(group_list)):
        u_list, _ = group_list[i]
        group = []
        for u in u_list:
            index = tree.nodes[u]['children'].tolist()
            group += index
        cluster_dist.update({k: i for k in group})
    results = sorted(cluster_dist.items(), key=lambda x: x[0])
    results = np.array([x[1] for x in results])
    return results


def merge_nodes_once(manifold, root_coords, tree, group_list, count):
    # group_list should be ordered ascend
    v1, v2 = group_list[-1], group_list[-2]
    merged_node = v1[0] + v2[0]
    merged_coords = torch.stack([tree.nodes[v]['coords'] for v in merged_node], dim=0)
    merged_point = manifold.Frechet_mean(merged_coords)
    merged_dist = manifold.dist(merged_point, root_coords).cpu().numpy()
    merged_item = (merged_node, merged_dist)
    del group_list[-2:]
    group_list.append(merged_item)
    group_list = sorted(group_list, reverse=False, key=lambda x: x[1])
    count -= 1
    return group_list, count


class cluster_metrics:
    def __init__(self, trues, predicts):
        self.true_label = trues
        self.pred_label = predicts

    def get_new_predicts(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0, 0, 0, 0, 0, 0, 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        self.new_predicts = new_predict

    def clusterAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0, 0, 0, 0, 0, 0, 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        self.new_predicts = new_predict
        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro

    def evaluateFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        ami = metrics.adjusted_mutual_info_score(self.true_label, self.pred_label)
        ari = metrics.adjusted_rand_score(self.true_label, self.pred_label)

        return nmi, ami, ari


def cal_AUC_AP(scores, trues):
    auc = metrics.roc_auc_score(trues, scores)
    ap = metrics.average_precision_score(trues, scores)
    return auc, ap



class Node:
    def __init__(self, index: list, embeddings: torch.Tensor, coords=None,
                 tree_index=None, is_leaf=False, height: int = None):
        self.index = index  # T_alpha
        self.embeddings = embeddings  # coordinates of nodes in T_alpha
        self.children = []
        self.coords = coords  # node coordinates
        self.tree_index = tree_index
        self.is_leaf = is_leaf
        self.height = height


def construct_tree(nodes_list: torch.LongTensor, manifold, coords_list: dict,
                   ass_list: dict, height, num_nodes):
    nodes_count = num_nodes
    que = Queue()
    root = Node(nodes_list, coords_list[height][nodes_list].cpu(),
                coords=coords_list[0].cpu(), tree_index=nodes_count, height=0)
    que.put(root)

    while not que.empty():
        node = que.get()
        L_nodes = node.index
        k = node.height + 1
        if k == height:
            for i in L_nodes:
                node.children.append(Node(i.reshape(-1), coords_list[height][i].cpu(), coords=coords_list[k][i].cpu(),
                                          tree_index=i.item(), is_leaf=True, height=k))
        else:
            temp_ass = ass_list[k][L_nodes].cpu()
            for j in range(temp_ass.shape[-1]):
                temp_child = L_nodes[temp_ass[:, j].nonzero().flatten()]
                if len(temp_child) > 0:
                    nodes_count += 1
                    child_node = Node(temp_child, coords_list[height][temp_child].cpu(),
                                      coords=coords_list[k][j].cpu(),
                                      tree_index=nodes_count, height=k)
                    node.children.append(child_node)
                    que.put(child_node)
    return root


def to_networkx_tree(root: Node, manifold, height):
    edges_list = []
    nodes_list = []
    que = Queue()
    que.put(root)
    nodes_list.append(
        (
            root.tree_index,
            {'coords': root.coords.reshape(-1),
             'is_leaf': root.is_leaf,
             'children': root.index,
             'height': root.height}
        )
    )

    while not que.empty():
        cur_node = que.get()
        if cur_node.height == height:
            break
        for node in cur_node.children:
            nodes_list.append(
                (
                    node.tree_index,
                    {'coords': node.coords.reshape(-1),
                     'is_leaf': node.is_leaf,
                     'children': node.index,
                     'height': node.height}
                )
            )
            edges_list.append(
                (
                    cur_node.tree_index,
                    node.tree_index,
                    {'weight': torch.sigmoid(1. - manifold.dist(cur_node.coords, node.coords)).item()}
                )
            )
            que.put(node)

    graph = nx.Graph()
    graph.add_nodes_from(nodes_list)
    graph.add_edges_from(edges_list)
    return graph



MIN_NORM = 1e-15
EPS = 1e-6


class HyperSE(nn.Module):
    # def __init__(self, args, manifold, n_layers, device, in_features, hidden_features, num_nodes, height=3, temperature=0.2,
    #              embed_dim=2, out_dim = 2, dropout=0.5, nonlin='relu', decay_rate=None, max_nums=None, use_att=True, use_bias=True):
    def __init__(self, args, manifold, n_layers, device, in_features, hidden_dim_enc, hidden_features, num_nodes, height=3, temperature=0.2,
                 embed_dim=2, dropout=0.5, nonlin='relu', decay_rate=None, max_nums=None, use_att=True, use_bias=True):
        
        super(HyperSE, self).__init__()
        self.num_nodes = num_nodes
        self.height = height
        self.tau = temperature
        self.manifold = manifold
        self.device = device
        self.encoder = LSENet(args, self.manifold, n_layers, in_features, hidden_dim_enc, hidden_features,
                              num_nodes, height, temperature, embed_dim, dropout,
                              nonlin, decay_rate, max_nums, use_att, use_bias)
        self.optimizer_pre = RiemannianAdam(self.parameters(), lr=args.lr_pre, weight_decay=args.w_decay)
        self.optimizer = RiemannianAdam(self.parameters(), lr=args.lr, weight_decay=args.w_decay)

    def forward(self, features, adj):
        features = features.to(self.device)
        adj = adj.to(self.device)
        adj = adj.to_dense()
        embeddings, clu_mat = self.encoder(features, adj)
        self.embeddings = {}
        self.num_nodes = features.shape[0]
        for height, x in embeddings.items():
            self.embeddings[height] = x.detach()
        ass_mat = {self.height: torch.eye(self.num_nodes).to(self.device)}
        for k in range(self.height - 1, 0, -1):
            ass_mat[k] = ass_mat[k + 1] @ clu_mat[k + 1]
        for k, v in ass_mat.items():
            idx = v.max(1)[1]
            t = torch.zeros_like(v)
            t[torch.arange(t.shape[0]), idx] = 1.
            ass_mat[k] = t
        self.ass_mat = ass_mat
        return self.embeddings[self.height]

    def loss(self, input_data: DSIData):

        device = input_data.device
        weight = input_data.weight.to(self.device)
        adj = input_data.adj.to(self.device)
        degrees = input_data.degrees.to(self.device)
        features = input_data.feature.to(self.device)
        edge_index = input_data.edge_index.to(self.device)
        neg_edge_index = input_data.neg_edge_index.to(self.device)
        pretrain = input_data.pretrain
        self.num_nodes = features.shape[0]

        embeddings, clu_mat = self.encoder(features, adj.to_dense())

        se_loss = 0
        vol_G = weight.sum()
        ass_mat = {self.height: torch.eye(self.num_nodes).to(self.device)}
        vol_dict = {self.height: degrees, 0: vol_G.unsqueeze(0)}
        for k in range(self.height - 1, 0, -1):
            ass_mat[k] = ass_mat[k + 1] @ clu_mat[k + 1]
            vol_dict[k] = torch.einsum('ij, i->j', ass_mat[k], degrees)

        edges = torch.cat([edge_index, neg_edge_index], dim=-1)
        prob = self.manifold.dist(embeddings[self.height][edges[0]], embeddings[self.height][edges[1]])
        prob = torch.sigmoid((2. - prob) / 1.)
        label = torch.cat([torch.ones(edge_index.shape[-1]), torch.zeros(neg_edge_index.shape[-1])]).to(self.device)
        lp_loss = F.binary_cross_entropy(prob, label)

        if pretrain:
            return self.manifold.dist0(embeddings[0]) + lp_loss

        for k in range(1, self.height + 1):
            vol_parent = torch.einsum('ij, j->i', clu_mat[k], vol_dict[k - 1])  # (N_k, )
            log_vol_ratio_k = torch.log2((vol_dict[k] + EPS) / (vol_parent + EPS))  # (N_k, )
            ass_i = ass_mat[k][edge_index[0]]   # (E, N_k)
            ass_j = ass_mat[k][edge_index[1]]
            weight_sum = torch.einsum('en, e->n', ass_i * ass_j, weight)  # (N_k, )
            delta_vol = vol_dict[k] - weight_sum    # (N_k, )
            se_loss += torch.sum(delta_vol * log_vol_ratio_k)
        se_loss = -1 / vol_G * se_loss
        return se_loss + self.manifold.dist0(embeddings[0]) + lp_loss
    


class LorentzGraphConvolution(nn.Module):
    """
    Hyperbolic graph convolution layer.
    """

    def __init__(self, manifold, in_features, out_features, use_bias, dropout, use_att, nonlin=None):
        super(LorentzGraphConvolution, self).__init__()
        self.linear = LorentzLinear(manifold, in_features, out_features, use_bias, dropout, nonlin=nonlin)
        self.agg = LorentzAgg(manifold, out_features, dropout, use_att)

    def forward(self, x, edge_index):
        h = self.linear(x)
        h = self.agg(h, edge_index)
        return h


class LorentzLinear(nn.Module):
    def __init__(self,
                 manifold,
                 in_features,
                 out_features,
                 bias=True,
                 dropout=0.1,
                 scale=10,
                 fixscale=False,
                 nonlin=None):
        super().__init__()
        self.manifold = manifold
        self.nonlin = nonlin
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.weight = nn.Linear(
            self.in_features, self.out_features, bias=bias)
        self.reset_parameters()
        self.dropout = nn.Dropout(dropout)
        self.scale = nn.Parameter(torch.ones(()) * math.log(scale), requires_grad=not fixscale)

    def forward(self, x):
        if self.nonlin is not None:
            x = self.nonlin(x)
        x = self.weight(self.dropout(x))
        x_narrow = x.narrow(-1, 1, x.shape[-1] - 1)
        time = x.narrow(-1, 0, 1).sigmoid() * self.scale.exp() + 1.1
        scale = (time * time - 1) / \
            (x_narrow * x_narrow).sum(dim=-1, keepdim=True).clamp_min(1e-8)
        x = torch.cat([time, x_narrow * scale.sqrt()], dim=-1)
        return x

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        step = self.in_features
        nn.init.uniform_(self.weight.weight, -stdv, stdv)
        with torch.no_grad():
            for idx in range(0, self.in_features, step):
                self.weight.weight[:, idx] = 0
        if self.bias:
            nn.init.constant_(self.weight.bias, 0)


class LorentzAgg(nn.Module):
    """
    Lorentz aggregation layer.
    """

    def __init__(self, manifold, in_features, dropout, use_att):
        super(LorentzAgg, self).__init__()
        self.manifold = manifold

        self.in_features = in_features
        self.dropout = dropout
        self.use_att = use_att
        if self.use_att:
            self.key_linear = LorentzLinear(manifold, in_features, in_features)
            self.query_linear = LorentzLinear(manifold, in_features, in_features)
            self.bias = nn.Parameter(torch.zeros(()) + 20)
            self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(in_features))

    def forward(self, x, adj):
        if self.use_att:
            query = self.query_linear(x)
            key = self.key_linear(x)
            att_adj = 2 + 2 * self.manifold.cinner(query, key)
            att_adj = att_adj / self.scale + self.bias
            att_adj = torch.sigmoid(att_adj)
            att_adj = torch.mul(adj.to_dense(), att_adj)
            support_t = torch.matmul(att_adj, x)
        else:
            support_t = torch.matmul(adj, x)

        denorm = (-self.manifold.inner(None, support_t, keepdim=True))
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        output = support_t / denorm
        return output


class LorentzAssignment(nn.Module):
    def __init__(self, manifold, in_features, hidden_features, num_assign, dropout,
                 bias=False, use_att=False, nonlin=None, temperature=0.2):
        super(LorentzAssignment, self).__init__()
        self.manifold = manifold
        self.num_assign = num_assign
        self.proj = nn.Sequential(LorentzLinear(manifold, in_features, hidden_features,
                                                     bias=bias, dropout=dropout, nonlin=None),
                                  # LorentzLinear(manifold, hidden_features, hidden_features,
                                  #               bias=bias, dropout=dropout, nonlin=nonlin)
                                  )
        self.assign_linear = LorentzGraphConvolution(manifold, hidden_features, num_assign + 1, use_att=use_att,
                                                     use_bias=bias, dropout=dropout, nonlin=nonlin)
        self.temperature = temperature
        self.key_linear = LorentzLinear(manifold, in_features, in_features)
        self.query_linear = LorentzLinear(manifold, in_features, in_features)
        self.bias = nn.Parameter(torch.zeros(()) + 20)
        self.scale = nn.Parameter(torch.zeros(()) + math.sqrt(hidden_features))

    def forward(self, x, adj):
        ass = self.assign_linear(self.proj(x), adj).narrow(-1, 1, self.num_assign)
        query = self.query_linear(x)
        key = self.key_linear(x)
        att_adj = 2 + 2 * self.manifold.cinner(query, key)
        att_adj = att_adj / self.scale + self.bias
        att = torch.sigmoid(att_adj)
        # att = torch.mul(adj.to_dense(), att)
        att = torch.mul(adj, att)
        ass = torch.matmul(att, ass)   # (N_k, N_{k-1})
        logits = torch.log_softmax(ass, dim=-1)
        return logits


class LSENetLayer(nn.Module):
    def __init__(self, manifold, in_features, hidden_features, num_assign, dropout,
                 bias=False, use_att=False, nonlin=None, temperature=0.2):
        super(LSENetLayer, self).__init__()
        self.manifold = manifold
        self.assignor = LorentzAssignment(manifold, in_features, hidden_features, num_assign, use_att=use_att, bias=bias,
                                          dropout=dropout, nonlin=nonlin, temperature=temperature)
        self.temperature = temperature

    def forward(self, x, adj):
        ass = self.assignor(x, adj)
        support_t = ass.exp().t() @ x
        denorm = (-self.manifold.inner(None, support_t, keepdim=True))
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        x_assigned = support_t / denorm
        adj = ass.exp().t() @ adj @ ass.exp()
        adj = adj - torch.eye(adj.shape[0]).to(adj.device) * adj.diag()
        adj = gumbel_sigmoid(adj, tau=self.temperature)
        return x_assigned, adj, ass.exp()



## K-Means in the Poincare Disk model

class PoincareKMeans(object):
    def __init__(self,n_clusters=8,n_init=20,max_iter=300,tol=1e-8,verbose=True):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose =  verbose
        self.labels_=None
        self.cluster_centers_ = None
           

    def fit(self,X):
        n_samples = X.shape[0]
        self.inertia = None

        for run_it in range(self.n_init):
            centroids = X[sample(range(n_samples),self.n_clusters),:]
            for it in range(self.max_iter):
                distances = self._get_distances_to_clusters(X,centroids)
                labels = np.argmin(distances,axis=1)

                new_centroids = np.zeros((self.n_clusters,2))
                for i in range(self.n_clusters):
                    indices = np.where(labels==i)[0]
                    if len(indices)>0:
                        new_centroids[i,:] = self._hyperbolic_centroid(X[indices,:])
                    else:
                        new_centroids[i,:] = X[sample(range(n_samples),1),:]
                m = np.ravel(centroids-new_centroids, order='K')
                diff = np.dot(m,m)
                centroids = new_centroids.copy()
                if(diff<self.tol):
                    break
                
            distances = self._get_distances_to_clusters(X,centroids)
            labels = np.argmin(distances,axis=1)
            inertia = np.sum([np.sum(distances[np.where(labels==i)[0],i]**2) for i in range(self.n_clusters)])
            if (self.inertia == None) or (inertia<self.inertia):
                self.inertia = inertia
                self.labels_ = labels.copy()
                self.cluster_centers_ = centroids.copy()
                
            if self.verbose:
                print("Iteration: {} - Best Inertia: {}".format(run_it,self.inertia))
                          
    def fit_predict(self,X):
        self.fit(X)
        return self.labels_

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
    
    def predict(self,X):
        distances = self.transform(X)
        return np.argmin(distances,axis=1)
        
    def transform(self,X):
        return self._get_distances_to_clusters(X,self.cluster_centers_)
    
    def _get_distances_to_clusters(self,X,clusters):
        n_samples,n_clusters = X.shape[0],clusters.shape[0]
        
        distances = np.zeros((n_samples,n_clusters))
        for i in range(n_clusters):
            centroid = np.tile(clusters[i,:],(n_samples,1))
            den1 = 1-np.linalg.norm(X,axis=1)**2
            den2 = 1-np.linalg.norm(centroid,axis=1)**2
            the_num = np.linalg.norm(X-centroid,axis=1)**2
            distances[:,i] = np.arccosh(1+2*the_num/(den1*den2))
        
        return distances
      
    def _poinc_to_minsk(self,points):
        minsk_points = np.zeros((points.shape[0],3))
        minsk_points[:,0] = np.apply_along_axis(arr=points,axis=1,func1d=lambda v: 2*v[0]/(1-v[0]**2-v[1]**2))
        minsk_points[:,1] = np.apply_along_axis(arr=points,axis=1,func1d=lambda v: 2*v[1]/(1-v[0]**2-v[1]**2))
        minsk_points[:,2] = np.apply_along_axis(arr=points,axis=1,func1d=lambda v: (1+v[0]**2+v[1]**2)/(1-v[0]**2-v[1]**2))
        return minsk_points

    def _minsk_to_poinc(self,points):
        poinc_points = np.zeros((points.shape[0],2))
        poinc_points[:,0] = points[:,0]/(1+points[:,2])
        poinc_points[:,1] = points[:,1]/(1+points[:,2])
        return poinc_points

    def _hyperbolic_centroid(self,points):
        minsk_points = self._poinc_to_minsk(points)
        minsk_centroid = np.mean(minsk_points,axis=0)
        normalizer = np.sqrt(np.abs(minsk_centroid[0]**2+minsk_centroid[1]**2-minsk_centroid[2]**2))
        minsk_centroid = minsk_centroid/normalizer
        return self._minsk_to_poinc(minsk_centroid.reshape((1,3)))[0]





class LSENet(nn.Module):
    def __init__(self, args, manifold, n_layers, in_features, hidden_dim_enc, hidden_features, num_nodes, height=3, temperature=0.1,
                 embed_dim=64, dropout=0.5, nonlin='relu', decay_rate=None, max_nums=None, use_att=True, use_bias=True):
        super(LSENet, self).__init__()
        if max_nums is not None:
            assert len(max_nums) == height - 1, "length of max_nums must equal height-1."
        self.args = args
        self.manifold = manifold
        self.nonlin = select_activation(nonlin) if nonlin is not None else None
        self.temperature = temperature
        self.num_nodes = num_nodes
        self.height = height
        self.scale = nn.Parameter(torch.tensor([0.999]), requires_grad=True)
        self.embed_layer = GraphEncoder(self.manifold, n_layers, in_features + 1, hidden_dim_enc, embed_dim + 1, 
                                        use_att=use_att, use_bias=use_bias, dropout=dropout, nonlin=self.nonlin)
        
        self.layers = nn.ModuleList([])
        if max_nums is None:
            decay_rate = int(np.exp(np.log(num_nodes) / height)) if decay_rate is None else decay_rate
            max_nums = [int(num_nodes / (decay_rate ** i)) for i in range(1, height)]
        for i in range(height - 1):
            self.layers.append(LSENetLayer(self.manifold, embed_dim + 1, hidden_features, max_nums[i],
                                           bias=use_bias, use_att=use_att, dropout=dropout,
                                           nonlin=self.nonlin, temperature=self.temperature))

    def forward(self, z, edge_index):

        if not self.args.hgae:
            o = torch.zeros_like(z).to(z.device)
            z = torch.cat([o[:, 0:1], z], dim=1)
            z = self.manifold.expmap0(z)
        z = self.embed_layer(z, edge_index)
        z = self.normalize(z)

        self.tree_node_coords = {self.height: z}
        self.assignments = {}

        edge = edge_index.clone()
        ass = None
        for i, layer in enumerate(self.layers):
            z, edge, ass = layer(z, edge)
            self.tree_node_coords[self.height - i - 1] = z
            self.assignments[self.height - i] = ass

        self.tree_node_coords[0] = self.manifold.Frechet_mean(z)
        self.assignments[1] = torch.ones(ass.shape[-1], 1).to(z.device)

        return self.tree_node_coords, self.assignments

    def normalize(self, x):
        x = self.manifold.to_poincare(x).to(self.scale.device)
        x = F.normalize(x, p=2, dim=-1) * self.scale.clamp(1e-2, 0.999)
        x = self.manifold.from_poincare(x)
        return x
    


class Lorentz(geoopt.Lorentz):
    def __init__(self, k=1.0, learnable=False):
        super(Lorentz, self).__init__(k, learnable)

    def cinner(self, x, y):
        x = x.clone()
        x.narrow(-1, 0, 1).mul_(-1)
        return x @ y.transpose(-1, -2)

    def to_poincare(self, x, dim=-1):
        x = x.to(self.device)
        dn = x.size(dim) - 1
        return x.narrow(dim, 1, dn) / (x.narrow(dim, 0, 1) + torch.sqrt(self.k))

    def from_poincare(self, x, dim=-1, eps=1e-6):
        x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
        res = (
                torch.sqrt(self.k)
                * torch.cat((1 + x_norm_square, 2 * x), dim=dim)
                / (1.0 - x_norm_square + eps)
        )
        return res

    def Frechet_mean(self, x, weights=None, keepdim=False):
        if weights is None:
            z = torch.sum(x, dim=0, keepdim=True)
        else:
            z = torch.sum(x * weights, dim=0, keepdim=keepdim)
        denorm = self.inner(None, z, keepdim=keepdim)
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        z = z / denorm
        return z


class Poincare(geoopt.PoincareBall):
    def __init__(self, c=1.0, learnable=False):
        super(Poincare, self).__init__(c=c, learnable=learnable)

    def from_lorentz(self, x, dim=-1):
        x = x.to(self.c.device)
        dn = x.size(dim) - 1
        return x.narrow(dim, 1, dn) / (x.narrow(dim, 0, 1) + torch.sqrt(self.c))

    def to_lorentz(self, x, dim=-1, eps=1e-6):
        x = x.to(self.c.device)
        x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
        res = (
                torch.sqrt(self.c)
                * torch.cat((1 + x_norm_square, 2 * x), dim=dim)
                / (1.0 - x_norm_square + eps)
        )
        return res

    def Frechet_mean(self, embeddings, weights=None, keepdim=False):
        z = self.to_lorentz(embeddings)
        if weights is None:
            z = torch.sum(z, dim=0, keepdim=True)
        else:
            z = torch.sum(z * weights, dim=0, keepdim=keepdim)
        denorm = lmath.inner(z, z, keepdim=keepdim)
        denorm = denorm.abs().clamp_min(1e-8).sqrt()
        z = z / denorm
        z = self.from_lorentz(z).to(embeddings.device)
        return z





def mask_edges(edge_index, neg_edges, val_prop=0.05, test_prop=0.1):
    n = len(edge_index[0])
    n_val = int(val_prop * n)
    n_test = int(test_prop * n)
    edge_val, edge_test, edge_train = edge_index[:, :n_val], edge_index[:, n_val:n_val + n_test], edge_index[:, n_val + n_test:]
    val_edges_neg, test_edges_neg = neg_edges[:, :n_val], neg_edges[:, n_val: n_test + n_val]
    train_edges_neg = torch.cat([neg_edges, val_edges_neg, test_edges_neg], dim=-1)
    if test_prop == 0:
        return (edge_train, edge_val), (train_edges_neg, val_edges_neg)
    else:
        return (edge_train, edge_val, edge_test), (train_edges_neg, val_edges_neg, test_edges_neg)


class TwitterDataSet:
    def __init__(self, args, dataset_name = "Event2012", block=None):

        self.args = args
        self.dataset_name = dataset_name

        self.path: str = f"../model/model_saved/hypersed/{self.dataset_name}/data"
        self.data = self.get_data(self.path)
        print('Got data.')


    def get_data(self, path):

        with open(path + f"/{self.args.encode}_embeddings.pkl", 'rb') as file:
            feature = pickle.load(file)

        temp = np.load(path + '/' + 'label.npy', allow_pickle=True)
        labels = np.asarray([int(each) for each in temp])
        num_classes = len(np.unique(labels))
        num_features = feature.shape[1]
        num_nodes = feature.shape[0]

        cur_path = f"{path}/message_graph_{self.args.edge_type}.npz"
        matrix = sparse.load_npz(cur_path)
        source_nodes, target_nodes = matrix.nonzero()
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        weight = torch.tensor(matrix.data, dtype=torch.float)
        degrees = scatter_sum(weight, edge_index[0])
        neg_edge_index = negative_sampling(edge_index)
        adj = index2adjacency(num_nodes, edge_index, weight, is_sparse=True)
        edge_index_type = EdgeIndexTypes(edge_index=edge_index, weight=weight, degrees=degrees, 
                                         neg_edge_index=neg_edge_index, adj=adj)
        
        if self.args.pre_anchor:
            anchor_matrix, anchor_fea, anchor_ass, anchor_labels = get_euc_anchors(feature, adj, self.args.anchor_rate, 
                                                                                   self.args.diag, labels)
            num_anchors = anchor_fea.shape[0]
            anchor_edge_index = torch.nonzero(anchor_matrix, as_tuple=False).t()
            # anchor_edge_index = torch.tensor([anchor_source_nodes, anchor_target_nodes], dtype=torch.long)
            anchor_weight = anchor_matrix[anchor_edge_index[0], anchor_edge_index[1]]
            anchor_degrees = scatter_sum(anchor_weight, anchor_edge_index[0])
            anchor_neg_edge_index = negative_sampling(anchor_edge_index)
            anchor_adj = index2adjacency(num_anchors, anchor_edge_index, anchor_weight, is_sparse=True)
            anchor_edge_index_type = EdgeIndexTypes(edge_index=anchor_edge_index, weight=anchor_weight, 
                                                    degrees=anchor_degrees, neg_edge_index=anchor_neg_edge_index, 
                                                    adj=anchor_adj)
        else:
            num_anchors, anchor_fea, anchor_edge_index_type, anchor_ass, anchor_labels = None, None, None, None, None
        
        return SingleBlockData(feature=feature, num_features=num_features, labels=labels, 
                               num_nodes=num_nodes, num_classes=num_classes, 
                               edge_index_types=edge_index_type, anchor_feature=anchor_fea, num_anchors=num_anchors,
                               anchor_edge_index_types=anchor_edge_index_type, anchor_ass=anchor_ass, anchor_labels=anchor_labels)


    def init_incr_data(self):

        for i in sorted(os.listdir(self.path), key=int):
            if i == '0':
                continue
        # for i in (19,20):    # for test
            self.datas.append(self.get_block_data(os.path.join(self.path, str(i))))



class GraphEncoder(nn.Module):
    def __init__(self, manifold, n_layers, in_features, n_hidden, out_dim,
                 dropout, nonlin=None, use_att=False, use_bias=False):
        super(GraphEncoder, self).__init__()
        self.manifold = manifold
        self.layers = nn.ModuleList([])
        self.layers.append(LorentzGraphConvolution(self.manifold, in_features,
                                                   n_hidden, use_bias, dropout=dropout, use_att=use_att, nonlin=None))
        for i in range(n_layers - 2):
            self.layers.append(LorentzGraphConvolution(self.manifold, n_hidden,
                                                       n_hidden, use_bias, dropout=dropout, use_att=use_att, nonlin=nonlin))
        self.layers.append(LorentzGraphConvolution(self.manifold, n_hidden,
                                                       out_dim, use_bias, dropout=dropout, use_att=use_att, nonlin=nonlin))

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
        return x


class FermiDiracDecoder(nn.Module):
    def __init__(self, 
                 args,
                 manifold):
        super(FermiDiracDecoder, self).__init__()
        
        self.args = args
        self.manifold = manifold
        self.r = self.args.r
        self.t = self.args.t

    def forward(self, x):
        
        N = x.shape[0]
        dist = torch.zeros((N, N), device=x.device)  # 初始化一个 N x N 的结果张量
        
        for i in range(N):
            # 计算第 i 行的所有距离
            dist[i, :] = self.manifold.dist2(x[i].unsqueeze(0), x)

        probs = torch.sigmoid((self.r - dist) / self.t)
        adj_pred = torch.sigmoid(probs)

        return adj_pred


class HyperGraphAutoEncoder(nn.Module):
    def __init__(self, args, device, manifold, n_layers, in_features, n_hidden, out_dim, dropout, nonlin, use_att, use_bias):
        super(HyperGraphAutoEncoder, self).__init__()

        self.args = args
        self.device = device
        self.manifold = manifold
        self.scale = nn.Parameter(torch.tensor([0.999]), requires_grad=True)

        self.encoder = GraphEncoder(self.manifold, n_layers, in_features + 1, n_hidden, out_dim + 1, 
                                    dropout, nonlin, use_att, use_bias)
        self.decoder = FermiDiracDecoder(self.args, self.manifold)
        self.optimizer = RiemannianAdam(self.parameters(), lr=self.args.lr_gae, weight_decay=args.w_decay)


    def forward(self, x, adj):
        x = x.to(self.device)
        adj = adj.to(self.device)
        
        o = torch.zeros_like(x).to(x.device)
        x = torch.cat([o[:, 0:1], x], dim=1)
        x = self.manifold.expmap0(x)
        z = self.encoder(x, adj)
        z = self.normalize(z)
        adj_pred = self.decoder(z)
        
        return adj_pred, z

    
    def loss(self, x, adj):
        x = x.to(self.device)
        
        # 获取预测值和隐藏表示
        adj_pred, z = self.forward(x, adj)
        
        # 确保预测值在[0,1]范围内
        adj_pred = torch.clamp(adj_pred, min=0.0, max=1.0)

        # 转换为稀疏矩阵并计算权重
        adj = tensor_to_sparse(adj, (adj.shape[0], adj.shape[1]))
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float(
            (adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
        
        # 创建标签
        adj_label = adj + sp.eye(adj.shape[0])
        adj_label = sparse_to_tuple(adj_label)
        adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
                                            torch.FloatTensor(adj_label[1]),
                                            torch.Size(adj_label[2]))
        
        # 转换为密集张量并确保值在[0,1]范围内
        adj_label_dense = adj_label.to_dense()
        adj_label_dense = torch.clamp(adj_label_dense, min=0.0, max=1.0)
        
        # 计算权重掩码
        weight_mask = adj_label_dense.view(-1) == 1
        weight_tensor = torch.ones(weight_mask.size(0))
        weight_tensor[weight_mask] = pos_weight
        
        # 移动到正确的设备
        adj_label_dense = adj_label_dense.to(self.device)
        weight_tensor = weight_tensor.to(self.device)
        
        # 计算损失
        loss = norm * F.binary_cross_entropy(
            adj_pred.view(-1),
            adj_label_dense.view(-1),
            weight=weight_tensor,
            reduction='mean'
        )
        
        return loss, adj_pred, z
    

    def normalize(self, x):
        x = self.manifold.to_poincare(x)
        x = x.to(self.device)
        x = F.normalize(x, p=2, dim=-1) * self.scale.clamp(1e-2, 0.999)
        x = self.manifold.from_poincare(x)
        return x




def select_activation(activation):
    if activation == 'elu':
        return F.elu
    elif activation == 'relu':
        return F.relu
    elif activation == 'sigmoid':
        return F.sigmoid
    elif activation is None:
        return None
    else:
        raise NotImplementedError('the non_linear_function is not implemented')

def Frechet_mean_poincare(manifold, embeddings, weights=None, keepdim=False):
    z = manifold.from_poincare(embeddings)
    if weights is None:
        z = torch.sum(z, dim=0, keepdim=True)
    else:
        z = torch.sum(z * weights, dim=0, keepdim=keepdim)
    denorm = manifold.inner(None, z, keepdim=keepdim)
    denorm = denorm.abs().clamp_min(1e-8).sqrt()
    z = z / denorm
    z = manifold.to_poincare(z).to(embeddings.device)
    return z

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature=1):
    y = logits + sample_gumbel(logits.size()).to(logits.device)
    return torch.nn.functional.softmax(y / temperature, dim=-1)

def gumbel_softmax(logits, temperature=0.2, hard=False):
    """
    ST-gumple-softmax
    input: [*, n_class]
    return: flatten --> [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)

    if not hard:
        return y

    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    # Set gradients w.r.t. y_hard gradients w.r.t. y
    y_hard = (y_hard - y).detach() + y
    return y_hard

def gumbel_sigmoid(logits, tau: float = 1, hard: bool = False, threshold: float = 0.5):
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0, 1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits, tau)
    y_soft = gumbels.sigmoid()

    if hard:
        # Straight through.
        indices = (y_soft > threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

def graph_top_K(dense_adj, k):
    assert k < dense_adj.shape[-1]
    _, indices = dense_adj.topk(k=k, dim=-1)
    mask = torch.zeros(dense_adj.shape).bool().to(dense_adj.device)
    mask[torch.arange(dense_adj.shape[0])[:, None], indices] = True
    sparse_adj = torch.masked_fill(dense_adj, ~mask, value=0.)
    return sparse_adj

def adjacency2index(adjacency, weight=False, topk=False, k=10):
    """_summary_

    Args:
        adjacency (torch.tensor): [N, N] matrix
    return:
        edge_index: [2, E]
        edge_weight: optional
    """
    if topk and k:
        adj = graph_top_K(adjacency, k)
    else:
        adj = adjacency
    edge_index = torch.nonzero(adj).t().contiguous()
    if weight:
        weight = adjacency[edge_index[0], edge_index[1]].reshape(-1)
        return edge_index, weight
    else:
        return edge_index

def index2adjacency(N, edge_index, weight=None, is_sparse=True):
    adjacency = torch.zeros(N, N).to(edge_index.device)
    m = edge_index.shape[1]
    if weight is None:
        adjacency[edge_index[0], edge_index[1]] = 1
    else:
        adjacency[edge_index[0], edge_index[1]] = weight.reshape(-1)
    if is_sparse:
        weight = weight if weight is not None else torch.ones(m).to(edge_index.device)
        adjacency = torch.sparse_coo_tensor(indices=edge_index, values=weight, size=(N, N))
    return adjacency

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def tensor_to_sparse(sparse_tensor, size = (500, 500)):
    # 获取稀疏张量的索引和值
    indices_np = sparse_tensor.coalesce().indices().numpy()
    values_np = sparse_tensor.coalesce().values().numpy()

    # 将索引转换为行和列
    row = indices_np[0]
    col = indices_np[1]

    # 创建 csr_matrix
    csr = csr_matrix((values_np, (row, col)), shape=size)
    return csr

def getOtherByedge(edge_index, num_nodes):
    weight = torch.ones(edge_index.shape[1])
    degrees = scatter_sum(weight, edge_index[0])
    adj = index2adjacency(num_nodes, edge_index, weight, is_sparse=True)

    return adj, degrees, weight

def getNewPredict(predicts, C):
    N = predicts.shape[0]
    P = np.zeros(N)
    C = C.cpu().numpy()
    for i in range(C.shape[0]):
        j = np.where(C[i] == 1)[0][0]
        P[i] = predicts[j]

    return P

def getC(Z, M):
    N = Z.size(0)
    Z_np = Z.detach().cpu().numpy()

    # 随机选择M个数据点作为初始锚点，并且确保每个聚类簇中至少有一个数据点
    initial_indices = np.random.choice(N, M, replace=False)
    initial_anchors = Z_np[initial_indices]

    # 对得到的特征进行kmeans聚类，使用初始锚点
    kmeans = KMeans(n_clusters=M, init=initial_anchors, n_init=1, max_iter=200, tol=1e-10)
    kmeans.fit(Z_np)
    labels = kmeans.labels_
    labels = torch.tensor(labels, device=Z.device, dtype=torch.long)

    C = torch.zeros(N, M, device=Z.device)
    C[torch.arange(N, dtype=torch.long), labels] = 1

    return C

def getNewPredict(predicts, C):
    P = np.zeros(C.shape[0])
    C = C.cpu().numpy()
    for i in range(C.shape[0]):
        j = np.where(C[i] == 1)[0][0]
        P[i] = predicts[j]

    return P

def get_anchor(Z, A, M):
    N = Z.size(0)
    Z_np = Z.detach().cpu().numpy()

    kmeans = PoincareKMeans(n_clusters=M, n_init=1, max_iter=200, tol=1e-10, verbose=True)
    kmeans.fit(Z_np)
    labels = kmeans.labels_
    labels = torch.tensor(labels, device=Z.device, dtype=torch.long)

    C = torch.zeros(N, M, device=Z.device)
    C[torch.arange(N, dtype=torch.long), labels] = 1

    # 计算锚点图的邻接矩阵
    A_anchor = C.T @ A @ C
    A_anchor.fill_diagonal_(0)

    # 计算锚点的表示
    X_anchor = torch.zeros(M, Z.size(1), device=Z.device)
    for i in range(M):
        cluster_points = Z[torch.where(C[:, i] == 1)]
        if cluster_points.size(0) > 0:  # 检查簇中是否存在数据点
            X_anchor[i] = cluster_points.mean(dim=0)

    return A_anchor, X_anchor, C

class Artanh(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.clamp(-1 + 1e-15, 1 - 1e-15)
        ctx.save_for_backward(x)
        z = x.double()
        return (torch.log_(1 + z).sub_(torch.log_(1 - z))).mul_(0.5).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / (1 - input ** 2)

def artanh(x):
    return Artanh.apply(x)

def tanh(x, clamp=15):
    return x.clamp(-clamp, clamp).tanh()



def mobius_add(x, y, c, dim=-1, eps=1e-5):
    x2 = x.pow(2).sum(dim=dim, keepdim=True)
    y2 = y.pow(2).sum(dim=dim, keepdim=True)
    xy = (x * y).sum(dim=dim, keepdim=True)
    num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
    denom = 1 + 2 * c * xy + c ** 2 * x2 * y2
    return num / denom.clamp_min(1e-15)

def hyperbolic_distance1(p1, p2, c=1):
    sqrt_c = c ** 0.5
    dist_c = artanh(
        sqrt_c * mobius_add(-p1, p2, c, dim=-1).norm(dim=-1, p=2, keepdim=False)
    )
    dist = dist_c * 2 / sqrt_c
    return dist ** 2

def hyperbolic_distance(z_u, z_h, eps=1e-5):
    norm_z_u = torch.norm(z_u, p=2, dim=-1)
    norm_z_h = torch.norm(z_h, p=2, dim=-1)

    # Ensure norms are less than 1 to satisfy the Poincaré ball constraint
    norm_z_u = torch.clamp(norm_z_u, max=1 - eps)
    norm_z_h = torch.clamp(norm_z_h, max=1 - eps)

    # Compute the squared Euclidean distance
    euclidean_dist_sq = torch.sum((z_u - z_h) ** 2, dim=-1)

    # Compute the hyperbolic distance
    numerator = 2 * euclidean_dist_sq
    denominator = (1 - norm_z_u ** 2) * (1 - norm_z_h ** 2)
    arg_acosh = 1 + numerator / denominator

    # Ensure the argument of acosh is >= 1
    arg_acosh = torch.clamp(arg_acosh, min=1 + eps)

    return torch.acosh(arg_acosh)


def contrastive_loss(manifold, z_u, z_h, z_h_all, temperature=0.5):

    dist1 = manifold.dist2(z_u, z_h)
    dist = manifold.dist2(z_u, z_h_all)

    loss = -torch.log(torch.exp(dist1 / temperature) / torch.exp(dist / temperature).sum())

    return loss.mean()


def L_ConV(manifold, z_u, z_h, z_e, N_s, temperature=0.5):
    """
    Compute the overall contrastive loss.
    """
    loss = 0.0
    for i in range(N_s):
        loss += (contrastive_loss(manifold, z_u[i], z_h[i], z_h, temperature) +
                 contrastive_loss(manifold, z_h[i], z_e[i], z_e, temperature) +
                 contrastive_loss(manifold, z_e[i], z_u[i], z_u, temperature))
    return loss / (3 * N_s)


def get_agg_feauture(manifold, x1, x2, x3):
    
    x = torch.zeros_like(x1)

    for i in range(x1.shape[0]):
        x_chunk = torch.stack((x1[i], x2[i], x3[i]), dim=0)  # (3, hidden)
        x[i] = manifold.Frechet_mean(x_chunk, keepdim=False)  # (hidden,)
    
    return x


def get_euc_anchors(features, adj, anchor_rate, diag, true_labels):

    num_nodes = features.shape[0]
    num_anchor = int(num_nodes / anchor_rate)

    kmeans = KMeans(n_clusters=num_anchor, n_init=10, random_state=1)
    anchor_result = kmeans.fit(features)
    anchor_predictions = anchor_result.labels_

    labels = get_cluster_labels(anchor_predictions, true_labels)

    anchor_predictions = torch.tensor(anchor_predictions, device=features.device, dtype=torch.long)

    C = torch.zeros(num_nodes, num_anchor, device=features.device)
    C[torch.arange(num_nodes, dtype=torch.long), anchor_predictions] = 1

    anchor_fea = torch.zeros(num_anchor, features.size(1), device=features.device)
    for i in range(num_anchor):
        cluster_points = features[torch.where(C[:, i] == 1)]
        if cluster_points.size(0) > 0:
            anchor_fea[i] = cluster_points.mean(dim=0)
    
    anchor_adj = C.T @ adj.to_dense() @ C
    # anchor_adj.fill_diagonal_(diag)

    return anchor_adj, anchor_fea, C, labels


def get_cluster_labels(cluster_predictions, true_labels):
    unique_clusters = set(cluster_predictions)
    cluster_labels = {}
    labels = []

    for cluster in unique_clusters:
        cluster_indices = [i for i, pred in enumerate(cluster_predictions) if pred == cluster]
        labels_in_cluster = [true_labels[idx] for idx in cluster_indices]
        most_common_label = Counter(labels_in_cluster).most_common(1)[0][0]
        cluster_labels[cluster] = most_common_label
        labels.append(most_common_label)

    unique_cluster_labels = set(cluster_labels.values())

    return labels


def get_euc_anchors_alladj(features, adj, anchor_rate, diag, thres):

    num_nodes = features.shape[0]
    num_anchor = int(num_nodes / anchor_rate)

    kmeans = KMeans(n_clusters=num_anchor, n_init=10, random_state=1)
    anchor_result = kmeans.fit(features)
    anchor_predictions = anchor_result.labels_

    anchor_predictions = torch.tensor(anchor_predictions, device=features.device, dtype=torch.long)

    C = torch.zeros(num_nodes, num_anchor, device=features.device)
    C[torch.arange(num_nodes, dtype=torch.long), anchor_predictions] = 1

    anchor_fea = torch.zeros(num_anchor, features.size(1), device=features.device)
    for i in range(num_anchor):
        cluster_points = features[torch.where(C[:, i] == 1)]
        if cluster_points.size(0) > 0:
            anchor_fea[i] = cluster_points.mean(dim=0)

    corr_matrix_np = np.corrcoef(features, rowvar=True)
    adjacency_matrix_np = np.where(corr_matrix_np > thres, corr_matrix_np, 0)
    adj = torch.tensor(adjacency_matrix_np, dtype=torch.float32)

    anchor_adj = C.T @ adj @ C
    # anchor_adj.fill_diagonal_(diag)

    return anchor_adj, anchor_fea, C


def get_euc_anchors_alladj_as(features, adj, anchor_rate, diag, thres):

    num_nodes = features.shape[0]
    num_anchor = int(num_nodes / anchor_rate)

    kmeans = KMeans(n_clusters=num_anchor, n_init=10, random_state=1)
    anchor_result = kmeans.fit(features)
    anchor_predictions = anchor_result.labels_

    anchor_predictions = torch.tensor(anchor_predictions, device=features.device, dtype=torch.long)

    C = torch.zeros(num_nodes, num_anchor, device=features.device)
    C[torch.arange(num_nodes, dtype=torch.long), anchor_predictions] = 1

    anchor_fea = torch.zeros(num_anchor, features.size(1), device=features.device)
    for i in range(num_anchor):
        cluster_points = features[torch.where(C[:, i] == 1)]
        if cluster_points.size(0) > 0:
            anchor_fea[i] = cluster_points.mean(dim=0)

    corr_matrix_np = np.corrcoef(features, rowvar=True)
    adjacency_matrix_np = np.where(corr_matrix_np > thres, corr_matrix_np, 0)
    adj = torch.tensor(adjacency_matrix_np + adj, dtype=torch.float32)

    anchor_adj = C.T @ adj @ C
    # anchor_adj.fill_diagonal_(diag)

    return anchor_adj, anchor_fea, C

def replaceAtUser(text):
    """ Replaces "@user" with "" """
    text = re.sub('@[^\s]+|RT @[^\s]+', '', text)
    return text


def removeUnicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)', r'', text)
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    return text


def replaceURL(text):
    """ Replaces url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'url', text)
    text = re.sub(r'#([^\s]+)', r'\1', text)
    return text


def replaceMultiExclamationMark(text):
    """ Replaces repetitions of exlamation marks """
    text = re.sub(r"(\!)\1+", '!', text)
    return text


def replaceMultiQuestionMark(text):
    """ Replaces repetitions of question marks """
    text = re.sub(r"(\?)\1+", '?', text)
    return text


def removeEmoticons(text):
    """ Removes emoticons from text """
    text = re.sub(
        ':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:',
        '', text)
    return text

def removeNewLines(text):
    text = re.sub('\n', '', text)
    return text


def preprocess_sentence(s):
    return removeNewLines(replaceAtUser(
        removeEmoticons(replaceMultiQuestionMark(replaceMultiExclamationMark(removeUnicode(replaceURL(s)))))))

def preprocess_french_sentence(s):
    return removeNewLines(
        replaceAtUser(removeEmoticons(replaceMultiQuestionMark(replaceMultiExclamationMark(replaceURL(s))))))

def SBERT_embed(s_list, language):
    '''
    Use Sentence-BERT to embed sentences.
    s_list: a list of sentences/ tokens to be embedded.
    language: the language of the sentences ('English', 'French', 'Arabic').
    output: the embeddings of the sentences/ tokens.
    '''
    # Model paths or names for each language
    model_map = {
        'English': '../model/model_needed/all-MiniLM-L6-v2',
        'French': '../model/model_needed/distiluse-base-multilingual-cased-v1',
        'Arabic': '../model/model_needed/paraphrase-multilingual-mpnet-base-v2'
    }

    # Default model for Hugging Face
    hf_model_map = {
        'English': 'sentence-transformers/all-MiniLM-L6-v2',
        'French': 'sentence-transformers/distiluse-base-multilingual-cased-v1',
        'Arabic': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
    }

    # Print language and model being used
    print(f"Embedding sentences in language: {language}")
    
    # Determine model path
    model_path = model_map.get(language)
    if not model_path:
        raise ValueError(f"Unsupported language: {language}. Supported languages are: {', '.join(model_map.keys())}")

    print(f"Using model: {model_path}")

    # Load the model, downloading if necessary
    try:
        model = SentenceTransformer(model_path)
        print(f"Successfully loaded model from local path: {model_path}")
    except Exception as e:
        print(f"Model {model_path} not found locally. Attempting to download from Hugging Face...")
        model = SentenceTransformer(hf_model_map[language])
        print(f"Model downloaded from Hugging Face: {hf_model_map[language]}")

    # Compute embeddings
    embeddings = model.encode(s_list, convert_to_tensor=True, normalize_embeddings=True)
    print(f"Computed embeddings for {len(s_list)} sentences/tokens.")
    
    return embeddings.cpu()


def compute_argmin(C, all_1dSEs):
    N = len(all_1dSEs)
    min_val = float('inf')
    min_i = None
    
    for j, i in enumerate(C):
        sum_val = 1/N * np.sum(all_1dSEs) - all_1dSEs[j]
        
        if sum_val < min_val:
            min_val = sum_val
            min_i = i
    
    return min_i

def search_threshold(embeddings, start=0.6, end=0.4, step=-0.05):
    all_1dSEs = []
    
    corr_matrix = np.corrcoef(embeddings)
    np.fill_diagonal(corr_matrix, 0)
    
    for i in tqdm(np.arange(start, end, step)):
        threshold = i
        edges = [(s, d, corr_matrix[s, d]) for s, d in np.ndindex(corr_matrix.shape) if corr_matrix[s, d] >= threshold]
        g = nx.Graph()
        g.add_weighted_edges_from(edges)
        seg = SE(g)
        all_1dSEs.append(seg.calc_1dSE())
    
    best_threshold = compute_argmin(np.arange(start, end, step), all_1dSEs)
    print('best threshold:', best_threshold)
    
    return best_threshold


class SE:
    def __init__(self, graph: nx.Graph):
        self.graph = graph.copy()
        self.vol = self.get_vol()
        self.division = {}  # {comm1: [node11, node12, ...], comm2: [node21, node22, ...], ...}
        self.struc_data = {}  # {comm1: [vol1, cut1, community_node_SE, leaf_nodes_SE], comm2:[vol2, cut2, community_node_SE, leaf_nodes_SE]，... }
        self.struc_data_2d = {} # {comm1: {comm2: [vol_after_merge, cut_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], comm3: [], ...}, ...}

    def get_vol(self):
        '''
        get the volume of the graph
        '''
        return cuts.volume(self.graph, self.graph.nodes, weight = 'weight')

    def calc_1dSE(self):
        '''
        get the 1D SE of the graph
        '''
        SE = 0
        for n in self.graph.nodes:
            d = cuts.volume(self.graph, [n], weight = 'weight')
            SE += - (d / self.vol) * math.log2(d / self.vol)
            #SE += - (d / self.vol) * log2(d / self.vol)
        return SE

    def update_1dSE(self, original_1dSE, new_edges):
        '''
        get the updated 1D SE after new edges are inserted into the graph
        '''
    
        affected_nodes = []
        for edge in new_edges:
            affected_nodes += [edge[0], edge[1]]
        affected_nodes = set(affected_nodes)

        original_vol = self.vol
        original_degree_dict = {node:0 for node in affected_nodes}
        for node in affected_nodes.intersection(set(self.graph.nodes)):
            original_degree_dict[node] = self.graph.degree(node, weight = 'weight')

        # insert new edges into the graph
        self.graph.add_weighted_edges_from(new_edges)

        self.vol = self.get_vol()
        updated_vol = self.vol
        updated_degree_dict = {}
        for node in affected_nodes:
            updated_degree_dict[node] = self.graph.degree(node, weight = 'weight')
        
        updated_1dSE = (original_vol / updated_vol) * (original_1dSE - math.log2(original_vol / updated_vol))
        for node in affected_nodes:
            d_original = original_degree_dict[node]
            d_updated = updated_degree_dict[node]
            if d_original != d_updated:
                if d_original != 0:
                    updated_1dSE += (d_original / updated_vol) * math.log2(d_original / updated_vol)
                updated_1dSE -= (d_updated / updated_vol) * math.log2(d_updated / updated_vol)

        return updated_1dSE

    def get_cut(self, comm):
        '''
        get the sum of the degrees of the cut edges of community comm
        '''
        return cuts.cut_size(self.graph, comm, weight = 'weight')

    def get_volume(self, comm):
        '''
        get the volume of community comm
        '''
        return cuts.volume(self.graph, comm, weight = 'weight')

    def calc_2dSE(self):
        '''
        get the 2D SE of the graph
        '''
        SE = 0
        for comm in self.division.values():
            g = self.get_cut(comm)
            v = self.get_volume(comm)
            SE += - (g / self.vol) * math.log2(v / self.vol)
            for node in comm:
                d = self.graph.degree(node, weight = 'weight')
                SE += - (d / self.vol) * math.log2(d / v)
        return SE

    def show_division(self):
        print(self.division)

    def show_struc_data(self):
        print(self.struc_data)
    
    def show_struc_data_2d(self):
        print(self.struc_data_2d)
        
    def print_graph(self):
        fig, ax = plt.subplots()
        nx.draw(self.graph, ax=ax, with_labels=True)
        plt.show()
    
    def update_struc_data(self):
        '''
        calculate the volume, cut, communitiy mode SE, and leaf nodes SE of each cummunity, 
        then store them into self.struc_data
        '''
        self.struc_data = {} # {comm1: [vol1, cut1, community_node_SE, leaf_nodes_SE], comm2:[vol2, cut2, community_node_SE, leaf_nodes_SE]，... }
        for vname in self.division.keys():
            comm = self.division[vname]
            volume = self.get_volume(comm)
            cut = self.get_cut(comm)
            if volume == 0:
                vSE = 0
            else:
                vSE = - (cut / self.vol) * math.log2(volume / self.vol)
            vnodeSE = 0
            for node in comm:
                d = self.graph.degree(node, weight = 'weight')
                if d != 0:
                    vnodeSE -= (d / self.vol) * math.log2(d / volume)
            self.struc_data[vname] = [volume, cut, vSE, vnodeSE]

    def update_struc_data_2d(self):
        '''
        calculate the volume, cut, communitiy mode SE, and leaf nodes SE after merging each pair of cummunities, 
        then store them into self.struc_data_2d
        '''
        self.struc_data_2d = {} # {(comm1, comm2): [vol_after_merge, cut_after_merge, comm_node_SE_after_merge, leaf_nodes_SE_after_merge], (comm1, comm3): [], ...}
        comm_num = len(self.division)
        for i in range(comm_num):
            for j in range(i + 1, comm_num):
                v1 = list(self.division.keys())[i]
                v2 = list(self.division.keys())[j]
                if v1 < v2:
                    k = (v1, v2)
                else:
                    k = (v2, v1)

                comm_merged = self.division[v1] + self.division[v2]
                gm = self.get_cut(comm_merged)
                vm = self.struc_data[v1][0] + self.struc_data[v2][0]
                if self.struc_data[v1][0] == 0 or self.struc_data[v2][0] == 0:
                    vmSE = self.struc_data[v1][2] + self.struc_data[v2][2]
                    vmnodeSE = self.struc_data[v1][3] + self.struc_data[v2][3]
                else:
                    vmSE = - (gm / self.vol) * math.log2(vm / self.vol)
                    vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0]/ self.vol) * math.log2(self.struc_data[v1][0] / vm) + \
                            self.struc_data[v2][3] - (self.struc_data[v2][0]/ self.vol) * math.log2(self.struc_data[v2][0] / vm)
                self.struc_data_2d[k] = [vm, gm, vmSE, vmnodeSE]

    def init_division(self):
        '''
        initialize self.division such that each node assigned to its own community
        '''
        self.division = {}
        for node in self.graph.nodes:
            new_comm = node
            self.division[new_comm] = [node]
            self.graph.nodes[node]['comm'] = new_comm

    def add_isolates(self):
        '''
        add any isolated nodes into graph
        '''
        all_nodes = list(chain(*list(self.division.values())))
        all_nodes.sort()
        edge_nodes = list(self.graph.nodes)
        edge_nodes.sort()
        if all_nodes != edge_nodes:
            for node in set(all_nodes)-set(edge_nodes):
                self.graph.add_node(node)

    def update_division_MinSE(self):
        '''
        greedily update the encoding tree to minimize 2D SE
        '''
        def Mg_operator(v1, v2):
            '''
            MERGE operator. It calculates the delta SE caused by mergeing communities v1 and v2, 
            without actually merging them, i.e., the encoding tree won't be changed
            '''
            v1SE = self.struc_data[v1][2] 
            v1nodeSE = self.struc_data[v1][3]

            v2SE = self.struc_data[v2][2]
            v2nodeSE = self.struc_data[v2][3]

            if v1 < v2:
                k = (v1, v2)
            else:
                k = (v2, v1)
            vm, gm, vmSE, vmnodeSE = self.struc_data_2d[k]
            delta_SE = vmSE + vmnodeSE - (v1SE + v1nodeSE + v2SE + v2nodeSE)
            return delta_SE

        # continue merging any two communities that can cause the largest decrease in SE, 
        # until the SE can't be further reduced
        while True: 
            comm_num = len(self.division)
            delta_SE = 99999
            vm1 = None
            vm2 = None
            for i in range(comm_num):
                for j in range(i + 1, comm_num):
                    v1 = list(self.division.keys())[i]
                    v2 = list(self.division.keys())[j]
                    new_delta_SE = Mg_operator(v1, v2)
                    if new_delta_SE < delta_SE:
                        delta_SE = new_delta_SE
                        vm1 = v1
                        vm2 = v2

            if delta_SE < 0:
                # Merge v2 into v1, and update the encoding tree accordingly
                for node in self.division[vm2]:
                    self.graph.nodes[node]['comm'] = vm1
                self.division[vm1] += self.division[vm2]
                self.division.pop(vm2)

                volume = self.struc_data[vm1][0] + self.struc_data[vm2][0]
                cut = self.get_cut(self.division[vm1])
                vmSE = - (cut / self.vol) * math.log2(volume / self.vol)
                vmnodeSE = self.struc_data[vm1][3] - (self.struc_data[vm1][0]/ self.vol) * math.log2(self.struc_data[vm1][0] / volume) + \
                        self.struc_data[vm2][3] - (self.struc_data[vm2][0]/ self.vol) * math.log2(self.struc_data[vm2][0] / volume)
                self.struc_data[vm1] = [volume, cut, vmSE, vmnodeSE]
                self.struc_data.pop(vm2)

                struc_data_2d_new = {}
                for k in self.struc_data_2d.keys():
                    if k[0] == vm2 or k[1] == vm2:
                        continue
                    elif k[0] == vm1 or k[1] == vm1:
                        v1 = k[0]
                        v2 = k[1]
                        comm_merged = self.division[v1] + self.division[v2]
                        gm = self.get_cut(comm_merged)
                        vm = self.struc_data[v1][0] + self.struc_data[v2][0]
                        if self.struc_data[v1][0] == 0 or self.struc_data[v2][0] == 0:
                            vmSE = self.struc_data[v1][2] + self.struc_data[v2][2]
                            vmnodeSE = self.struc_data[v1][3] + self.struc_data[v2][3]
                        else:
                            vmSE = - (gm / self.vol) * math.log2(vm / self.vol)
                            vmnodeSE = self.struc_data[v1][3] - (self.struc_data[v1][0]/ self.vol) * math.log2(self.struc_data[v1][0] / vm) + \
                                    self.struc_data[v2][3] - (self.struc_data[v2][0]/ self.vol) * math.log2(self.struc_data[v2][0] / vm)
                        struc_data_2d_new[k] = [vm, gm, vmSE, vmnodeSE]
                    else:
                        struc_data_2d_new[k] = self.struc_data_2d[k]
                self.struc_data_2d = struc_data_2d_new
            else:
                break


def get_graph_edges(attributes):
    attr_nodes_dict = {}
    for i, l in enumerate(attributes):
        for attr in l:
            if attr not in attr_nodes_dict:
                attr_nodes_dict[attr] = [i] # node indexing starts from 1
            else:
                attr_nodes_dict[attr].append(i)

    for attr in attr_nodes_dict.keys():
        attr_nodes_dict[attr].sort()

    graph_edges = []
    for l in attr_nodes_dict.values():
        graph_edges += list(combinations(l, 2))
    return list(set(graph_edges))

def get_knn_edges(embeddings, best_threshold):
    corr_matrix = np.corrcoef(embeddings)
    np.fill_diagonal(corr_matrix, 0)
    knn_edges = [(s, d, corr_matrix[s, d]) for s, d in np.ndindex(corr_matrix.shape) if corr_matrix[s, d] >= best_threshold]
        
    return list(set(knn_edges))

def get_global_edges(attributes, embeddings, best_threshold, e_a = True, e_s = True):
    graph_edges, knn_edges = [], []
    if e_a == True:
        graph_edges = get_graph_edges(attributes)
    if e_s == True:
        knn_edges = get_knn_edges(embeddings, best_threshold)
    return list(set(knn_edges + graph_edges))




    