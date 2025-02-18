import argparse
import json
import numpy as np
import os
from time import strftime, localtime, time
import torch
from scipy import sparse
from torch.utils.data import Dataset
import pandas as pd
from datetime import datetime
import spacy
import networkx as nx
import dgl
from dgl.data.utils import save_graphs, load_graphs
import pickle
from collections import Counter
import torch.optim as optim
from sklearn import metrics
from sklearn.cluster import KMeans
import torch.nn.functional as F
import torch.nn as nn
from itertools import combinations
import re
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataloader import Event2012,Arabic_Twitter,Event2018





class QSGNN:
    r"""The QSGNN model for social event detection that uses a query-based streaming graph neural network
    for event detection.

    .. note::
        This detector uses graph neural networks with query-based streaming to identify events in social media data.
        The model requires a dataset object with a load_data() method.

    Parameters
    ----------
    dataset : object
        The dataset object containing social media data.
        Must provide load_data() method that returns the raw data.
    finetune_epochs : int, optional
        Number of fine-tuning epochs. Default: ``1``.
    n_epochs : int, optional
        Number of training epochs. Default: ``5``.
    oldnum : int, optional
        Number of old classes. Default: ``20``.
    novelnum : int, optional
        Number of novel classes. Default: ``20``.
    n_infer_epochs : int, optional
        Number of inference epochs. Default: ``0``.
    window_size : int, optional
        Size of sliding window. Default: ``3``.
    patience : int, optional
        Early stopping patience. Default: ``5``.
    margin : float, optional
        Margin for triplet loss. Default: ``3.0``.
    a : float, optional
        Scaling factor. Default: ``8.0``.
    lr : float, optional
        Learning rate for optimizer. Default: ``1e-3``.
    batch_size : int, optional
        Batch size for training. Default: ``1000``.
    n_neighbors : int, optional
        Number of neighbors to sample. Default: ``1200``.
    word_embedding_dim : int, optional
        Word embedding dimension. Default: ``300``.
    hidden_dim : int, optional
        Hidden layer dimension. Default: ``16``.
    out_dim : int, optional
        Output dimension. Default: ``64``.
    num_heads : int, optional
        Number of attention heads. Default: ``4``.
    use_residual : bool, optional
        Whether to use residual connections. Default: ``True``.
    validation_percent : float, optional
        Percentage of data for validation. Default: ``0.1``.
    test_percent : float, optional
        Percentage of data for testing. Default: ``0.2``.
    use_hardest_neg : bool, optional
        Whether to use hardest negative mining. Default: ``True``.
    metrics : str, optional
        Evaluation metric to use. Default: ``'nmi'``.
    use_cuda : bool, optional
        Whether to use GPU acceleration. Default: ``True``.
    add_ort : bool, optional
        Whether to add orthogonal regularization. Default: ``True``.
    gpuid : int, optional
        GPU device ID to use. Default: ``0``.
    mask_path : str, optional
        Path to mask file. Default: ``None``.
    log_interval : int, optional
        Number of steps between logging. Default: ``10``.
    is_incremental : bool, optional
        Whether to use incremental learning. Default: ``True``.
    data_path : str, optional
        Path to save model data. Default: ``'../model/model_saved/qsgnn/English'``.
    file_path : str, optional
        Path to save model files. Default: ``'../model/model_saved/qsgnn'``.
    add_pair : bool, optional
        Whether to add pair-wise constraints. Default: ``False``.
    initial_lang : str, optional
        Initial language for processing. Default: ``'English'``.
    is_static : bool, optional
        Whether to use static graph. Default: ``False``.
    graph_lang : str, optional
        Language for graph construction. Default: ``'English'``.
    days : int, optional
        Number of days for temporal window. Default: ``2``.
    """
    
    def __init__(
        self,
        dataset,
        finetune_epochs=1,
        n_epochs=5,
        oldnum=20,
        novelnum=20,
        n_infer_epochs=0,
        window_size=3,
        patience=5,
        margin=3.0,
        a=8.0,
        lr=1e-3,
        batch_size=1000,
        n_neighbors=1200,
        word_embedding_dim=300,
        hidden_dim=16,
        out_dim=64,
        num_heads=4,
        use_residual=True,
        validation_percent=0.1,
        test_percent=0.2,
        use_hardest_neg=True,
        metrics='nmi',
        use_cuda=True,
        add_ort=True,
        gpuid=0,
        mask_path=None,
        log_interval=10,
        is_incremental=True,
        data_path='../model/model_saved/qsgnn/English',
        file_path='../model/model_saved/qsgnn',
        add_pair=False,
        initial_lang='English',
        is_static=False,
        graph_lang='English',
        days=2
    ):
        # 将参数赋值给 self
        self.finetune_epochs = finetune_epochs
        self.n_epochs = n_epochs
        self.oldnum = oldnum
        self.novelnum = novelnum
        self.n_infer_epochs = n_infer_epochs
        self.window_size = window_size
        self.patience = patience
        self.margin = margin
        self.a = a
        self.lr = lr
        self.batch_size = batch_size
        self.n_neighbors = n_neighbors
        self.word_embedding_dim = word_embedding_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.use_residual = use_residual
        self.validation_percent = validation_percent
        self.test_percent = test_percent
        self.use_hardest_neg = use_hardest_neg
        self.metrics = metrics
        self.use_cuda = use_cuda
        self.add_ort = add_ort
        self.gpuid = gpuid
        self.mask_path = mask_path
        self.log_interval = log_interval
        self.is_incremental = is_incremental
        self.data_path = data_path
        self.file_path = file_path
        self.add_pair = add_pair
        self.initial_lang = initial_lang
        self.is_static = is_static
        self.graph_lang = graph_lang
        self.days = days
        
        self.dataset = dataset.load_data()
        if self.use_cuda:
            torch.cuda.set_device(self.gpuid)

        self.data_split = None
        

    def preprocess(self):
        args=self
        preprocessor = Preprocessor(self.dataset)
        preprocessor.generate_initial_features(self.dataset)
        preprocessor.construct_graph()

        self.embedding_save_path = self.data_path + '/embeddings'
        os.makedirs(self.embedding_save_path, exist_ok=True)
        # with open(self.embedding_save_path + '/args.txt', 'w') as f:
        #     json.dump(self.__dict__, f, indent=2)
        self.data_split = np.load(self.data_path + '/data_split.npy')

    def fit(self):
        args=self
        if self.use_hardest_neg:
            loss_fn = OnlineTripletLoss(self.margin, HardestNegativeTripletSelector(self.margin))
        else:
            loss_fn = OnlineTripletLoss(self.margin, RandomNegativeTripletSelector(self.margin))
        metrics = [AverageNonzeroTripletsMetric()]

        if self.add_pair:
            self.model = GAT(302, self.hidden_dim, self.out_dim, self.num_heads, self.use_residual)
            best_model_path = self.embedding_save_path + '/block_0/models/best.pt'
            label_center_emb = torch.load(self.embedding_save_path + '/block_0/models/center.pth')
            self.model.load_state_dict(torch.load(best_model_path))

            if self.use_cuda:
                self.model.cuda()

            if self.is_incremental:
                kmeans_scores = []
                for i in range(1, self.data_split.shape[0]):
                    print("incremental setting")
                    print("enter i ", str(i))
                    _, score = continue_train(i, self.data_split, metrics, self.embedding_save_path, loss_fn,
                                              self.model, label_center_emb, args)
                    kmeans_scores.append(score)
                    print("KMeans:")
                    print_scores(kmeans_scores)
                print(self.finetune_epochs, self.oldnum, self.novelnum, self.a,
                      self.batch_size, end="\n\n")
        else:
            self.model = initial_train(0, self, self.data_split, metrics, self.embedding_save_path, loss_fn, None)
        print("fit:", type(self.model))

        torch.save(self.model.state_dict(), self.embedding_save_path + '/final_model.pth')

    def detection(self):
        data = SocialDataset(self.data_path, 0)
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        in_feats = features.shape[1]  # feature dimension

        g = dgl.DGLGraph(data.matrix, readonly=True)
        g.set_n_initializer(dgl.init.zero_initializer)
        g.readonly(readonly_state=True)

        predictions = []
        ground_truths = []
        self.detection_path = self.file_path + '/detection_split/'
        os.makedirs(self.detection_path, exist_ok=True)

        self.model = GAT(in_feats, self.hidden_dim, self.out_dim, self.num_heads, self.use_residual)
        best_model_path = self.embedding_save_path + '/block_0/models/best.pt'
        self.model.load_state_dict(torch.load(best_model_path))

        train_indices, validation_indices, test_indices = generateMasks(len(labels), self.data_split, 0,
                                                                        self.validation_percent,
                                                                        self.test_percent,
                                                                        self.detection_path)

        device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
        if args.use_cuda:
            self.model.cuda()  # 转移模型到cuda
            print("Model moved to CUDA.")
            g = g.to(device)
            features, labels = features.cuda(), labels.cuda()
            test_indices = test_indices.cuda()

        g.ndata['features'] = features
        g.ndata['labels'] = labels
        print("detection:", type(self.model))

        '''
        print("detection:", type(self.model))
        self.model = GAT(302, self.hidden_dim, self.out_dim, self.num_heads, self.use_residual)
        self.model.load_state_dict(torch.load(self.embedding_save_path + '/final_model.pth'))
        print("detection2:", type(self.model))
        '''

        extract_features, extract_labels = extract_embeddings(g, self.model, len(labels), self)

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


class Preprocessor(QSGNN):
    def __init__(self,dataser):
        super().__init__(dataset=dataset)

    def generate_initial_features(self,dataset):
        args=self
        save_path = args.file_path + '/features/'
        os.makedirs(save_path, exist_ok=True)

        df = dataset
        print(type(df))
        print("Loaded {} data  shape {}".format(args.initial_lang, df.shape))
        print(df.head(10))

        t_features = self.df_to_t_features(df)
        print("Time features generated.")
        d_features = self.documents_to_features(df, args.initial_lang)
        print("Original document features generated")

        combined_features = np.concatenate((d_features, t_features), axis=1)
        print("Concatenated document features and time features.")
        np.save(save_path + 'features_69612_0709_spacy_lg_zero_multiclasses_filtered_{}.npy'.format(args.initial_lang),
                combined_features)

    def documents_to_features(self, df, initial_lang):
        if initial_lang == "French":
            nlp = spacy.load('fr_core_news_lg')
        elif initial_lang == "Arabic":
            nlp = spacy.load('spacy.arabic.model')
            nlp.tokenizer = Arabic_preprocessor(nlp.tokenizer)
        elif initial_lang == "English":
            nlp = spacy.load('en_core_web_lg')
        else:
            print("not have that language!")
            return None

        features = df.filtered_words.apply(lambda x: nlp(' '.join(x)).vector if len(x) != 0 else nlp(' ').vector).values
        print(features)
        return np.stack(features, axis=0)

    def extract_time_feature(self, t_str):
        t = datetime.fromisoformat(str(t_str))
        OLE_TIME_ZERO = datetime(1899, 12, 30)
        delta = t - OLE_TIME_ZERO
        return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]  # 86,400 seconds in day

    def df_to_t_features(self, df):
        t_features = np.asarray([self.extract_time_feature(t_str) for t_str in df['created_at']])
        return t_features

    def construct_graph(self):
        args=self
        if args.is_static:
            save_path = "../model/model_saved/qsgnn/hash_static-{}-{}/".format(str(args.days), args.graph_lang)
        else:
            save_path = "../model/model_saved/qsgnn/{}/".format(args.graph_lang)

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        if args.graph_lang == "French":
             df = Event2018().load_data()
        elif args.graph_lang == "Arabic":
            df = Arabic_Twitter().load_data()
            name2id = {}
            for id, name in enumerate(df['event_id'].unique()):
                name2id[name] = id
            print(name2id)
            df['event_id'] = df['event_id'].apply(lambda x: name2id[x])
            df.drop_duplicates(['tweet_id'], inplace=True, keep='first')

        elif args.graph_lang == "English":
            df = Event2012().load_data()

        print("{} Data converted to dataframe.".format(args.graph_lang))
        df = df.sort_values(by='created_at').reset_index()

        df['created_at'] = pd.to_datetime(df['created_at'])
        df['date'] = [d.date() for d in df['created_at']]

        f = np.load(args.file_path + '/features/features_69612_0709_spacy_lg_zero_multiclasses_filtered_{}.npy'.format(
            args.graph_lang))

        message, data_split, all_graph_mins = self.construct_incremental_dataset(args, df, save_path, f, False)
        with open(save_path + "node_edge_statistics.txt", "w") as text_file:
            text_file.write(message)
        np.save(save_path + 'data_split.npy', np.asarray(data_split))
        print("Data split: ", data_split)
        np.save(save_path + 'all_graph_mins.npy', np.asarray(all_graph_mins))
        print("Time spent on heterogeneous -> homogeneous graph conversions: ", all_graph_mins)

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

            hashtags = row['hashtags']
            G.add_nodes_from(hashtags)
            for each in hashtags:
                G.nodes[each]['hashtag'] = True

            edges = []
            edges += [(tid, each) for each in user_ids]
            edges += [(tid, each) for each in entities]
            edges += [(tid, each) for each in hashtags]
            G.add_edges_from(edges)

        return G

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
        message += '\tDone. Time elapsed: ' + str(mins) + ' mins\n'

        print('\tGetting adjacency matrix ...')
        message += '\tGetting adjacency matrix ...\n'
        start = time()
        A = nx.to_numpy_array(G)  # Returns the graph adjacency matrix as a NumPy matrix.
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: ' + str(mins) + ' mins\n'

        print('\tGetting lists of nodes of various types ...')
        message += '\tGetting lists of nodes of various types ...\n'
        start = time()
        tid_nodes = list(nx.get_node_attributes(G, 'tweet_id').keys())
        userid_nodes = list(nx.get_node_attributes(G, 'user_id').keys())
        hash_nodes = list(nx.get_node_attributes(G, 'hashtag').keys())
        entity_nodes = list(nx.get_node_attributes(G, 'entity').keys())
        del G
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: ' + str(mins) + ' mins\n'

        print('\tConverting node lists to index lists ...')
        message += '\tConverting node lists to index lists ...\n'
        start = time()
        indices_tid = [all_nodes.index(x) for x in tid_nodes]
        indices_userid = [all_nodes.index(x) for x in userid_nodes]
        indices_hashtag = [all_nodes.index(x) for x in hash_nodes]
        indices_entity = [all_nodes.index(x) for x in entity_nodes]
        del tid_nodes
        del userid_nodes
        del hash_nodes
        del entity_nodes
        mins = (time() - start) / 60
        print('\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\tDone. Time elapsed: ' + str(mins) + ' mins\n'

        print('\tStart constructing tweet-user-tweet commuting matrix ...')
        print('\t\t\tStart constructing tweet-user matrix ...')
        message += '\tStart constructing tweet-user-tweet commuting matrix ...\n\t\t\tStart constructing tweet-user matrix ...\n'
        start = time()
        w_tid_userid = A[np.ix_(indices_tid, indices_userid)]
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'

        print('\t\t\tConverting to sparse matrix ...')
        message += '\t\t\tConverting to sparse matrix ...\n'
        start = time()
        s_w_tid_userid = sparse.csr_matrix(w_tid_userid)
        del w_tid_userid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'

        print('\t\t\tTransposing ...')
        message += '\t\t\tTransposing ...\n'
        start = time()
        s_w_userid_tid = s_w_tid_userid.transpose()
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'

        print('\t\t\tCalculating tweet-user * user-tweet ...')
        message += '\t\t\tCalculating tweet-user * user-tweet ...\n'
        start = time()
        s_m_tid_userid_tid = s_w_tid_userid * s_w_userid_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'

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
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'

        print('\tStart constructing tweet-ent-tweet commuting matrix ...')
        print('\t\t\tStart constructing tweet-ent matrix ...')
        message += '\tStart constructing tweet-ent-tweet commuting matrix ...\n\t\t\tStart constructing tweet-ent matrix ...\n'
        start = time()
        w_tid_entity = A[np.ix_(indices_tid, indices_entity)]
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'

        print('\t\t\tConverting to sparse matrix ...')
        message += '\t\t\tConverting to sparse matrix ...\n'
        start = time()
        s_w_tid_entity = sparse.csr_matrix(w_tid_entity)
        del w_tid_entity
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'

        print('\t\t\tTransposing ...')
        message += '\t\t\tTransposing ...\n'
        start = time()
        s_w_entity_tid = s_w_tid_entity.transpose()
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'

        print('\t\t\tCalculating tweet-ent * ent-tweet ...')
        message += '\t\t\tCalculating tweet-ent * ent-tweet ...\n'
        start = time()
        s_m_tid_entity_tid = s_w_tid_entity * s_w_entity_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'

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
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'

        print('\tStart constructing tweet-word-tweet commuting matrix ...')
        print('\t\t\tStart constructing tweet-word matrix ...')
        message += '\tStart constructing tweet-word-tweet commuting matrix ...\n\t\t\tStart constructing tweet-word matrix ...\n'
        start = time()
        w_tid_word = A[np.ix_(indices_tid, indices_hashtag)]
        del A
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'

        print('\t\t\tConverting to sparse matrix ...')
        message += '\t\t\tConverting to sparse matrix ...\n'
        start = time()
        s_w_tid_word = sparse.csr_matrix(w_tid_word)
        del w_tid_word
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'

        print('\t\t\tTransposing ...')
        message += '\t\t\tTransposing ...\n'
        start = time()
        s_w_word_tid = s_w_tid_word.transpose()
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'

        print('\t\t\tCalculating tweet-word * word-tweet ...')
        message += '\t\t\tCalculating tweet-word * word-tweet ...\n'
        start = time()
        s_m_tid_word_tid = s_w_tid_word * s_w_word_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'

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
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'

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
        s_bool_A_tid_tid = (s_A_tid_tid + s_m_tid_word_tid).astype('bool')
        del s_m_tid_word_tid
        del s_A_tid_tid
        mins = (time() - start) / 60
        print('\t\t\tDone. Time elapsed: ', mins, ' mins\n')
        message += '\t\t\tDone. Time elapsed: ' + str(mins) + ' mins\n'
        all_mins = (time() - all_start) / 60
        print('\tOver all time elapsed: ', all_mins, ' mins\n')
        message += '\tOver all time elapsed: ' + str(all_mins) + ' mins\n'

        if save_path is not None:
            sparse.save_npz(save_path + "s_bool_A_tid_tid.npz", s_bool_A_tid_tid)
            print("Sparse binary adjacency matrix saved.")
            s_bool_A_tid_tid = sparse.load_npz(save_path + "s_bool_A_tid_tid.npz")
            print("Sparse binary adjacency matrix loaded.")

        G = dgl.DGLGraph(s_bool_A_tid_tid)
        print('We have %d nodes.' % G.number_of_nodes())
        print('We have %d edges.' % G.number_of_edges())
        message += 'We have ' + str(G.number_of_nodes()) + ' nodes. We have ' + str(G.number_of_edges()) + ' edges.\n'

        return all_mins, message

    def construct_incremental_dataset(self, args, df, save_path, features, test=False):
        data_split = []
        all_graph_mins = []
        message = ""
        distinct_dates = df.date.unique()
        print("Number of distinct dates: ", len(distinct_dates))
        message += "Number of distinct dates: " + str(len(distinct_dates)) + "\n"
        print("Start constructing initial graph ...")
        message += "\nStart constructing initial graph ...\n"

        if args.is_static:
            ini_df = df.loc[df['date'].isin(distinct_dates[:args.days])]
        else:
            ini_df = df.loc[df['date'].isin(distinct_dates[:7])]

        path = save_path + '0/'
        if not os.path.exists(path):
            os.mkdir(path)

        y = ini_df['event_id'].values
        y = [int(each) for each in y]
        np.save(path + 'labels.npy', np.asarray(y))

        G = self.construct_graph_from_df(ini_df)
        grap_mins, graph_message = self.networkx_to_dgl_graph(G, save_path=path)
        message += graph_message
        print("Initial graph saved")
        message += "Initial graph saved\n"
        data_split.append(ini_df.shape[0])
        all_graph_mins.append(grap_mins)
        y = ini_df['event_id'].values
        y = [int(each) for each in y]
        np.save(path + 'labels.npy', np.asarray(y))
        np.save(path + 'df.npy', ini_df)
        print("Labels saved.")
        message += "Labels saved.\n"
        indices = ini_df['index'].values.tolist()
        x = features[indices, :]
        np.save(path + 'features.npy', x)
        print("Features saved.")
        message += "Features saved.\n\n"

        if not args.is_static:
            inidays = 7
            j = 6
            for i in range(inidays, len(distinct_dates)):
                print("Start constructing graph ", str(i - j), " ...")
                message += "\nStart constructing graph " + str(i - j) + " ...\n"
                incr_df = df.loc[df['date'] == distinct_dates[i]]
                path = save_path + str(i - j) + '/'
                if not os.path.exists(path):
                    os.mkdir(path)
                np.save(path + "/" + "dataframe.npy", incr_df)

                G = self.construct_graph_from_df(incr_df)
                grap_mins, graph_message = self.networkx_to_dgl_graph(G, save_path=path)
                message += graph_message
                print("Graph ", str(i - j), " saved")
                message += "Graph " + str(i - j) + " saved\n"
                data_split.append(incr_df.shape[0])
                all_graph_mins.append(grap_mins)
                y = [int(each) for each in incr_df['event_id'].values]
                np.save(path + 'labels.npy', y)
                print("Labels saved.")
                message += "Labels saved.\n"
                indices = incr_df['index'].values.tolist()
                x = features[indices, :]
                np.save(path + 'features.npy', x)
                np.save(path + 'df.npy', incr_df)
                print("Features saved.")
                message += "Features saved.\n"
        return message, data_split, all_graph_mins


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
            self.matrix = self.matrix[indices_to_keep, :]
            self.matrix = self.matrix[:, indices_to_keep]


class Arabic_preprocessor:
    def __init__(self, tokenizer, **cfg):
        self.tokenizer = tokenizer

    def clean_text(self, text):
        search = ["أ", "إ", "آ", "ة", "_", "-", "/", ".", "،", " و ", " يا ", '"', "ـ", "'", "ى", "\\", '\n', '\t',
                  '&quot;', '?', '؟', '!']
        replace = ["ا", "ا", "ا", "ه", " ", " ", "", "", "", " و", " يا", "", "", "", "ي", "", ' ', ' ', ' ', ' ? ',
                   ' ؟ ', ' ! ']

        # remove tashkeel
        p_tashkeel = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
        text = re.sub(p_tashkeel, "", text)

        # remove longation
        p_longation = re.compile(r'(.)\1+')
        subst = r"\1\1"
        text = re.sub(p_longation, subst, text)

        text = text.replace('وو', 'و')
        text = text.replace('يي', 'ي')
        text = text.replace('اا', 'ا')

        for i in range(len(search)):
            text = text.replace(search[i], replace[i])

        # trim    
        text = text.strip()

        return text

    def __call__(self, text):
        preprocessed = self.clean_text(text)
        return self.tokenizer(preprocessed)

class EDNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, use_dropout=False):
        super(EDNN, self).__init__()
        self.use_dropout = use_dropout
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        if self.use_dropout:
            hidden = F.dropout(hidden, training=self.training)
        out = self.fc2(hidden)
        return out


class simNN(nn.Module):
    def __init__(self, in_dim, use_dropout=False):
        super(simNN, self).__init__()
        self.fc = nn.Linear(in_dim, in_dim)

    def forward(self, x):
        hidden = self.fc(x)
        out = torch.mm(hidden, x.t())
        return out


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual=False):
        super(GATLayer, self).__init__()
        # equation (1) reference: https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/9_gat.html
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.use_residual = use_residual
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, blocks, layer_id):
        h = blocks[layer_id].srcdata['features']
        z = self.fc(h)
        blocks[layer_id].srcdata['z'] = z
        z_dst = z[:blocks[layer_id].number_of_dst_nodes()]

        blocks[layer_id].dstdata['z'] = z_dst
        blocks[layer_id].apply_edges(self.edge_attention)
        # equation (3) & (4)
        blocks[layer_id].update_all(  # block_id – The block to run the computation.
            self.message_func,  # Message function on the edges.
            self.reduce_func)  # Reduce function on the node.

        # nf.layers[layer_id].data.pop('z')
        # nf.layers[layer_id + 1].data.pop('z')

        if self.use_residual:
            return z_dst + blocks[layer_id].dstdata['h']  # residual connection
        return blocks[layer_id].dstdata['h']


class MultiHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', use_residual=False):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(in_dim, out_dim, use_residual))
        self.merge = merge

    def forward(self, blocks, layer_id):
        head_outs = [attn_head(blocks, layer_id) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=1)
            return torch.cat(head_outs, dim=1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads, use_residual=False):
        super(GAT, self).__init__()
        self.layer1 = MultiHeadGATLayer(in_dim, hidden_dim, num_heads, 'cat', use_residual)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(hidden_dim * num_heads, out_dim, 1, 'cat', use_residual)

    def forward(self, blocks):
        h = self.layer1(blocks, 0)
        h = F.elu(h)
        # print(h.shape)
        blocks[1].srcdata['features'] = h
        h = self.layer2(blocks, 1)
        # h = F.normalize(h, p=2, dim=1)
        return h


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


def generateMasks(length, data_split, i, validation_percent=0.2, test_percent=0.2, save_path=None):
    # verify total number of nodes
    print(length, data_split[i])
    assert length == data_split[i]
    if i == 0:
        # randomly suffle the graph indices
        train_indices = torch.randperm(length)
        # get total number of validation indices
        n_validation_samples = int(length * validation_percent)
        # sample n_validation_samples validation indices and use the rest as training indices
        validation_indices = train_indices[:n_validation_samples]
        n_test_samples = n_validation_samples + int(length * test_percent)
        test_indices = train_indices[n_validation_samples:n_test_samples]
        train_indices = train_indices[n_test_samples:]

        if save_path is not None:
            torch.save(validation_indices, save_path + '/validation_indices.pt')
            torch.save(train_indices, save_path + '/train_indices.pt')
            torch.save(test_indices, save_path + '/test_indices.pt')
            validation_indices = torch.load(save_path + '/validation_indices.pt')
            train_indices = torch.load(save_path + '/train_indices.pt')
            test_indices = torch.load(save_path + '/test_indices.pt')
        return train_indices, validation_indices, test_indices
    # If is in inference(prediction) epochs, generate test indices
    else:
        test_indices = torch.range(0, (data_split[i] - 1), dtype=torch.long)
        if save_path is not None:
            torch.save(test_indices, save_path + '/test_indices.pt')
            test_indices = torch.load(save_path + '/test_indices.pt')
        return test_indices


def getdata(embedding_save_path, data_split, i, args):
    save_path_i = embedding_save_path + '/block_' + str(i)
    if not os.path.isdir(save_path_i):
        os.mkdir(save_path_i)
    # load data
    data = SocialDataset(args.data_path, i)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    in_feats = features.shape[1]  # feature dimension

    g = dgl.DGLGraph(data.matrix,
                     readonly=True)
    num_isolated_nodes = graph_statistics(g, save_path_i)
    g.set_n_initializer(dgl.init.zero_initializer)
    g.readonly(readonly_state=True)

    mask_path = save_path_i + '/masks'
    if not os.path.isdir(mask_path):
        os.mkdir(mask_path)

    if i == 0:
        train_indices, validation_indices, test_indices = generateMasks(len(labels), data_split, i,
                                                                        args.validation_percent,
                                                                        args.test_percent,
                                                                        mask_path)
    else:
        test_indices = generateMasks(len(labels), data_split, i, args.validation_percent,
                                     args.test_percent,
                                     mask_path)
    device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
    if args.use_cuda:
        g = g.to(device)
        features, labels = features.cuda(), labels.cuda()
        test_indices = test_indices.cuda()
        if i == 0:
            train_indices, validation_indices = train_indices.cuda(), validation_indices.cuda()

    g.ndata['features'] = features
    g.ndata['labels'] = labels

    if i == 0:
        return save_path_i, in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices
    else:
        return save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices


def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3


def run_kmeans(extract_features, extract_labels, indices, args, isoPath=None):
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

    # kmeans clustering
    kmeans = KMeans(n_clusters=n_classes, random_state=0).fit(X)
    labels = kmeans.labels_
    nmi = metrics.normalized_mutual_info_score(labels_true, labels)
    ari = metrics.adjusted_rand_score(labels_true, labels)
    ami = metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic')
    print("nmi:", nmi, 'ami:', ami, 'ari:', ari)
    value = nmi
    global NMI
    NMI = nmi
    global AMI
    AMI = ami
    global ARI
    ARI = ari

    if args.metrics == 'ari':
        print('use ari')
        value = ari
    if args.metrics == 'ami':
        print('use ami')
        value = ami
    # Return number  of test tweets, number of classes covered by the test tweets, and kMeans cluatering NMI
    return (n_test_tweets, n_classes, value)


def evaluate(extract_features, extract_labels, indices, epoch, num_isolated_nodes, save_path, args, is_validation=True):
    message = ''
    message += '\nEpoch '
    message += str(epoch + 1)
    message += '\n'

    # with isolated nodes
    n_tweets, n_classes, value = run_kmeans(extract_features, extract_labels, indices, args)
    if is_validation:
        mode = 'validation'
    else:
        mode = 'test'
    message += '\tNumber of ' + mode + ' tweets: '
    message += str(n_tweets)
    message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
    message += str(n_classes)
    message += '\n\t' + mode + ' '
    message += args.metrics + ': '
    message += str(value)
    if num_isolated_nodes != 0:
        # without isolated nodes
        message += '\n\tWithout isolated nodes:'
        n_tweets, n_classes, value = run_kmeans(extract_features, extract_labels, indices, args,
                                                save_path + '/isolated_nodes.pt')
        message += '\tNumber of ' + mode + ' tweets: '
        message += str(n_tweets)
        message += '\n\tNumber of classes covered by ' + mode + ' tweets: '
        message += str(n_classes)
        message += '\n\t' + mode + ' value: '
        message += str(value)
    message += '\n'
    global NMI
    global AMI
    global ARI
    with open(save_path + '/evaluate.txt', 'a') as f:
        f.write(message)
        f.write('\n')
        f.write("NMI " + str(NMI) + " AMI " + str(AMI) + ' ARI ' + str(ARI))
    print(message)

    all_value_save_path = "/".join(save_path.split('/')[0:-1])
    print(all_value_save_path)

    with open(all_value_save_path + '/evaluate.txt', 'a') as f:
        f.write("block " + save_path.split('/')[-1])
        f.write(message)
        f.write('\n')
        f.write("NMI " + str(NMI) + " AMI " + str(AMI) + ' ARI ' + str(ARI) + '\n')

    return value, NMI, AMI, ARI


# 调整batch_size之后的函数
def extract_embeddings(g, model, num_all_samples, args):
    with torch.no_grad():
        model.eval()
        indices = torch.LongTensor(np.arange(0, num_all_samples, 1))
        if args.use_cuda:
            indices = indices.cuda()
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        batch_size = min(args.batch_size, num_all_samples)  # 使用较小的批量大小

        dataloader = dgl.dataloading.NodeDataLoader(
            g, block_sampler=sampler,
            batch_size=batch_size,
            nids=indices,
            shuffle=False,
            drop_last=False,
        )

        all_features = []
        all_labels = []

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]

            with torch.cuda.amp.autocast():  # 使用混合精度
                extract_labels = blocks[-1].dstdata['labels']
                extract_features = model(blocks)

            all_features.append(extract_features.data.cpu().numpy())
            all_labels.append(extract_labels.data.cpu().numpy())

            # 清理缓存
            del extract_features, extract_labels, blocks
            torch.cuda.empty_cache()

        extract_features = np.concatenate(all_features, axis=0)
        extract_labels = np.concatenate(all_labels, axis=0)

    return extract_features, extract_labels


def initial_train(i, args, data_split, metrics, embedding_save_path, loss_fn, model=None):
    print("Starting initial_train function.")
    save_path_i, in_feats, num_isolated_nodes, g, labels, train_indices, validation_indices, test_indices = getdata(
        embedding_save_path, data_split, i, args)

    print("Data loaded.")

    if model is None:  # Construct the initial model
        model = GAT(in_feats, args.hidden_dim, args.out_dim, args.num_heads, args.use_residual)
        print("Model constructed.")
    if args.use_cuda:
        model.cuda()
        print("Model moved to CUDA.")

    # Optimizer
    # optimizer = optim.Adam([{"params":model.parameters()},lr=args.lr, weight_decay=1e-4)
    optimizer = optim.Adam([{"params": model.parameters(), "lr": args.lr, "weight_decay": 1e-4}])
    print("Optimizer initialized.")

    # Start training
    message = "\n------------ Start initial training ------------\n"
    print(message)
    with open(save_path_i + '/log.txt', 'a') as f:
        f.write(message)
    # record the highest validation nmi ever got for early stopping
    best_vali_value = 1e-9
    best_epoch = 0
    wait = 0
    # record validation nmi of all epochs before early stop
    all_vali_value = []
    # record the time spent in seconds on each batch of all training/maintaining epochs
    seconds_train_batches = []
    # record the time spent in mins on each epoch
    mins_train_epochs = []
    for epoch in range(args.n_epochs):
        start_epoch = time()
        losses = []
        total_loss = 0
        for metric in metrics:
            metric.reset()

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        label_center = {}
        for l in set(extract_labels):
            l_indices = np.where(extract_labels == l)[0]
            l_feas = extract_features[l_indices]
            l_cen = np.mean(l_feas, 0)
            label_center[l] = l_cen

        print(f"Epoch {epoch + 1}/{args.n_epochs} - Features and labels extracted.")

        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        # dataloader = dgl.dataloading.NodeDataLoader(
        #     g, train_indices, sampler,
        #     batch_size=args.batch_size,
        #     shuffle=True,
        #     drop_last=False,
        #     )
        dataloader = dgl.dataloading.NodeDataLoader(
            g, block_sampler=sampler,
            batch_size=args.batch_size,
            nids=train_indices,
            shuffle=False,
            drop_last=False,
        )

        print(f"Epoch {epoch + 1}/{args.n_epochs} - DataLoader initialized.")

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            blocks = [b.to(device) for b in blocks]
            batch_labels = blocks[-1].dstdata['labels']

            start_batch = time()
            model.train()
            # forward
            pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow).

            print(f"Epoch {epoch + 1}/{args.n_epochs}, Batch {batch_id + 1} - Forward pass done.")

            # 计算到中心点的距离
            dis = torch.empty([0, 1]).cuda()
            for l in set(batch_labels.cpu().data.numpy()):
                label_indices = torch.where(batch_labels == l)
                l_center = torch.FloatTensor(label_center[l]).cuda()
                dis_l = (pred[label_indices] - l_center).pow(2).sum(1).unsqueeze(-1)
                dis = torch.cat([dis, dis_l], 0)

            if args.add_pair:
                pairs, pair_labels, pair_matrix = pairwise_sample(pred, batch_labels)
                if args.use_cuda:
                    pairs = pairs.to(device)
                    pair_matrix = pair_matrix.to(device)
                    pair_labels = pair_labels.to(device)

                pos_indices = torch.where(pair_labels > 0)
                neg_indices = torch.where(pair_labels == 0)
                neg_ind = torch.randint(0, neg_indices[0].shape[0], [5 * pos_indices[0].shape[0]]).to(device)
                neg_dis = (pred[pairs[neg_indices[0][neg_ind], 0]] - pred[pairs[neg_indices[0][neg_ind], 1]]).pow(
                    2).sum(1).unsqueeze(-1)
                pos_dis = (pred[pairs[pos_indices[0], 0]] - pred[pairs[pos_indices[0], 1]]).pow(2).sum(1).unsqueeze(-1)
                pos_dis = torch.cat([pos_dis] * 5, 0)
                pairs_indices = torch.where(torch.clamp(pos_dis + args.a - neg_dis, min=0.0) > 0)
                loss = torch.mean(torch.clamp(pos_dis + args.a - neg_dis, min=0.0)[pairs_indices[0]])

                label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda()
                pred = F.normalize(pred, 2, 1)
                pair_out = torch.mm(pred, pred.t())
                if args.add_ort:
                    pair_loss = (pair_matrix - pair_out).pow(2).mean()
                    print("pair loss:", loss, "pair orthogonal loss:  ", 100 * pair_loss)
                    loss += 100 * pair_loss
            else:
                # 使用 triplet loss 作为默认的损失函数
                loss_outputs = loss_fn(pred, batch_labels)
                loss = loss_outputs[0] if type(loss_outputs) in (tuple, list) else loss_outputs

            losses.append(loss.item())
            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}/{args.n_epochs}, Batch {batch_id + 1} - Backward pass and optimization done.")

            batch_seconds_spent = time() - start_batch
            seconds_train_batches.append(batch_seconds_spent)
            # end one batch

        total_loss /= (batch_id + 1)
        message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.n_epochs, total_loss)
        for metric in metrics:
            message += '\t{}: {:.4f}'.format(metric.name(), metric.value())
        mins_spent = (time() - start_epoch) / 60
        message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
        message += '\n'
        print(message)
        with open(save_path_i + '/log.txt', 'a') as f:
            f.write(message)
        mins_train_epochs.append(mins_spent)

        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        np.save(save_path_i + '/features_' + str(epoch) + '.npy', extract_features)
        np.save(save_path_i + '/labels_' + str(epoch) + '.npy', extract_labels)

        validation_value, _, _, _ = evaluate(extract_features, extract_labels, validation_indices, epoch,
                                             num_isolated_nodes,
                                             save_path_i, args, True)
        all_vali_value.append(validation_value)

        print(f"Epoch {epoch + 1}/{args.n_epochs} - Validation done. Value: {validation_value}")

        # Early stop
        if validation_value > best_vali_value:
            best_vali_value = validation_value
            best_epoch = epoch
            wait = 0
            # Save model
            model_path = save_path_i + '/models'
            if (epoch == 0) and (not os.path.isdir(model_path)):
                os.mkdir(model_path)
            p = model_path + '/best.pt'
            torch.save(model.state_dict(), p)
            print(f"Epoch {epoch + 1}/{args.n_epochs} - Best model saved.")

        else:
            wait += 1
        if wait == args.patience:
            print('Saved all_mins_spent')
            print('Early stopping at epoch ', str(epoch))
            print('Best model was at epoch ', str(best_epoch))
            break
        # end one epoch

    # Save all validation nmi
    np.save(save_path_i + '/all_vali_value.npy', np.asarray(all_vali_value))
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

    extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
    label_center = {}
    for l in set(extract_labels):
        l_indices = np.where(extract_labels == l)[0]
        l_feas = extract_features[l_indices]
        l_cen = np.mean(l_feas, 0)
        label_center[l] = l_cen
    label_center_emb = torch.FloatTensor(np.array(list(label_center.values()))).cuda()
    torch.save(label_center_emb, save_path_i + '/models/center.pth')
    print("Label center embeddings saved.")

    if args.add_pair:
        return model, label_center_emb
    else:
        return model


def continue_train(i, data_split, metrics, embedding_save_path, loss_fn, model, label_center_emb, args):
    save_path_i, in_feats, num_isolated_nodes, g, labels, test_indices = getdata(
        embedding_save_path, data_split, i, args)

    if i % 1 != 0:
        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
        # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
        test_value = evaluate(extract_features, extract_labels, test_indices, 0, num_isolated_nodes,
                              save_path_i, args, True)
        return model

    else:
        extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)

        _, nmi, ami, ari = evaluate(extract_features, extract_labels, test_indices, -1, num_isolated_nodes,
                                    save_path_i, args, True)
        score = {"NMI": nmi, "AMI": ami, "ARI": ari}
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        # Start fine tuning
        if i < 21:
            message = "\n------------ Start fine tuning ------------\n"
            print(message)
            with open(save_path_i + '/log.txt', 'a') as f:
                f.write(message)

            # record the time spent in seconds on each batch of all training/maintaining epochs
            seconds_train_batches = []
            # record the time spent in mins on each epoch
            mins_train_epochs = []
            for epoch in range(args.finetune_epochs):
                start_epoch = time()
                losses = []
                total_loss = 0
                for metric in metrics:
                    metric.reset()

                sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
                # dataloader = dgl.dataloading.NodeDataLoader(
                #     g, test_indices, sampler,
                #     batch_size=args.batch_size,
                #     shuffle=True,
                #     drop_last=False,
                #     )
                dataloader = dgl.dataloading.NodeDataLoader(
                    g, block_sampler=sampler,
                    batch_size=args.batch_size,
                    nids=test_indices,
                    shuffle=False,
                    drop_last=False,
                )

                for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                    device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
                    blocks = [b.to(device) for b in blocks]
                    batch_labels = blocks[-1].dstdata['labels']

                    start_batch = time()
                    model.train()
                    label_center_emb.to(device)

                    # forward
                    pred = model(blocks)  # Representations of the sampled nodes (in the last layer of the NodeFlow).
                    pred = F.normalize(pred, 2, 1)
                    rela_center_vec = torch.mm(pred, label_center_emb.t())
                    rela_center_vec = F.normalize(rela_center_vec, 2, 1)
                    entropy = torch.mul(torch.log(rela_center_vec), rela_center_vec)
                    entropy = torch.sum(entropy, dim=1)
                    value, old_indices = torch.topk(entropy.reshape(-1), int(entropy.shape[0] / 2), largest=True)
                    value, novel_indices = torch.topk(entropy.reshape(-1), int(entropy.shape[0] / 2), largest=False)
                    print(old_indices.shape, novel_indices.shape)
                    pair_matrix = torch.mm(rela_center_vec, rela_center_vec.t())

                    pairs, pair_labels, _ = pairwise_sample(F.normalize(pred, 2, 1), batch_labels)

                    if args.use_cuda:
                        pairs.cuda()
                        pair_labels.cuda()
                        pair_matrix.cuda()
                        # initial_pair_matrix.cuda()
                        model.cuda()

                    neg_values, novel_neg_ind = torch.topk(pair_matrix[novel_indices],
                                                           min(args.novelnum, pair_matrix[old_indices].size(0)), 1,
                                                           largest=False)
                    pos_values, novel_pos_ind = torch.topk(pair_matrix[novel_indices],
                                                           min(args.novelnum, pair_matrix[old_indices].size(0)), 1,
                                                           largest=True)
                    neg_values, old_neg_ind = torch.topk(pair_matrix[old_indices],
                                                         min(args.oldnum, pair_matrix[old_indices].size(0)), 1,
                                                         largest=False)
                    pos_values, old_pos_ind = torch.topk(pair_matrix[old_indices],
                                                         min(args.oldnum, pair_matrix[old_indices].size(0)), 1,
                                                         largest=True)

                    old_row = torch.LongTensor(
                        [[i] * min(args.oldnum, pair_matrix[old_indices].size(0)) for i in old_indices])
                    old_row = old_row.reshape(-1).cuda()
                    novel_row = torch.LongTensor(
                        [[i] * min(args.novelnum, pair_matrix[old_indices].size(0)) for i in novel_indices])
                    novel_row = novel_row.reshape(-1).cuda()
                    row = torch.cat([old_row, novel_row])
                    neg_ind = torch.cat([old_neg_ind.reshape(-1), novel_neg_ind.reshape(-1)])
                    pos_ind = torch.cat([old_pos_ind.reshape(-1), novel_pos_ind.reshape(-1)])
                    neg_distances = (pred[row] - pred[neg_ind]).pow(2).sum(1).unsqueeze(-1)
                    pos_distances = (pred[row] - pred[pos_ind]).pow(2).sum(1).unsqueeze(-1)

                    loss = torch.mean(torch.clamp(pos_distances + args.a - neg_distances, min=0.0))

                    losses.append(loss.item())
                    total_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_seconds_spent = time() - start_batch
                    seconds_train_batches.append(batch_seconds_spent)
                    # end one batch

                total_loss /= (batch_id + 1)
                message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(epoch + 1, args.finetune_epochs, total_loss)
                mins_spent = (time() - start_epoch) / 60
                message += '\nThis epoch took {:.2f} mins'.format(mins_spent)
                message += '\n'
                print(message)
                with open(save_path_i + '/log.txt', 'a') as f:
                    f.write(message)
                mins_train_epochs.append(mins_spent)

                extract_features, extract_labels = extract_embeddings(g, model, len(labels), args)
                # save_embeddings(extract_nids, extract_features, extract_labels, extract_train_tags, save_path_i, epoch)
                test_value, _, _, _ = evaluate(extract_features, extract_labels, test_indices, epoch,
                                               num_isolated_nodes,
                                               save_path_i, args, True)

            # Save model
            model_path = save_path_i + '/models'
            if not os.path.isdir(model_path):
                os.mkdir(model_path)
            p = model_path + '/finetune.pt'
            torch.save(model.state_dict(), p)
            print('finetune model saved after epoch ', str(epoch))

            # Save time spent on epochs
            np.save(save_path_i + '/mins_train_epochs.npy', np.asarray(mins_train_epochs))
            print('Saved mins_train_epochs.')
            # Save time spent on batches
            np.save(save_path_i + '/seconds_train_batches.npy', np.asarray(seconds_train_batches))
            print('Saved seconds_train_batches.')

        return model, score


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


def print_scores(scores):
    line = [' ' * 4] + [f'   M{i:02d} ' for i in range(1, len(scores) + 1)]
    print("".join(line))

    score_names = ['NMI', 'AMI', 'ARI']
    for n in score_names:
        line = [f'{n} '] + [f'  {s[n]:1.3f}' for s in scores]
        print("".join(line))
    print('\n', flush=True)

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


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
            torch.lgamma(sum_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device):
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    # evidence = relu_evidence(output)
    # alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device)
    )
    return loss


def edl_log_loss(alpha, target, epoch_num, num_classes, annealing_step, device):
    # evidence = relu_evidence(output)
    # alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
        alpha, target, epoch_num, num_classes, annealing_step, device):
    # evidence = relu_evidence(output)
    # alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def pairwise_sample(embeddings, labels=None, model=None):
    if model == None:  # labels is not None:
        labels = labels.cpu().data.numpy()
        indices = np.arange(0, len(labels), 1)
        pairs = np.array(list(combinations(indices, 2)))
        pair_labels = (labels[pairs[:, 0]] == labels[pairs[:, 1]])

        pair_matrix = np.eye(len(labels))
        ind = np.where(pair_labels)
        pair_matrix[pairs[ind[0], 0], pairs[ind[0], 1]] = 1
        pair_matrix[pairs[ind[0], 1], pairs[ind[0], 0]] = 1

        return torch.LongTensor(pairs), torch.LongTensor(pair_labels.astype(int)), torch.LongTensor(pair_matrix)

    else:
        pair_matrix = model(embeddings)
        return pair_matrix
    # torch.LongTensor(pair_labels.astype(int))


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


