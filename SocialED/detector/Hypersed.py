import os
import torch
import pickle
import numpy as np
import pandas as pd
import json
import re
import math
import matplotlib.pyplot as plt
from os.path import exists
from datetime import datetime
from tqdm import tqdm
from scipy import sparse
from itertools import combinations, chain
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from networkx.algorithms import cuts
import networkx as nx
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Hypersed():
    def __init__(self, dataset):
        self.dataset = dataset
        self.language = dataset.get_dataset_language()
        self.dataset_name = dataset.get_dataset_name()
        self.save_path = "../model/model_saved/hypersed/"+self.dataset_name+"/"

    def preprocess(self):
        preprocessor = Preprocessor(self.dataset)
        preprocessor.preprocess()

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



class Preprocessor:
    def __init__(self, dataset, mode='close'):
        """Initialize preprocessor
        Args:
            dataset: Dataset object (e.g. Event2012, Event2018, or other datasets)
            mode: 'open' or 'close' (default 'close') - determines preprocessing mode
        """
        self.dataset = dataset
        self.dataset_name = dataset.get_dataset_name()
        self.language = dataset.get_dataset_language()
        self.mode = mode
        self.base_path = '../model/model_saved/hypersed/'+self.dataset_name+'/'
        self.columns = ['tweet_id', 'text', 'event_id', 'words', 'filtered_words',
                       'entities', 'user_id', 'created_at', 'urls', 'hashtags', 'user_mentions']


    def preprocess(self):
        """Main preprocessing function"""
        df = self.dataset.load_data()
        print(f"Loaded {self.dataset_name} dataset")

        if self.mode == 'open':
            # Open-set processing
            open_path = f'{self.base_path}/open_set/'
            self.split_open_set(df, open_path)
            self.get_open_set_messages_embeddings()
            self.construct_open_set_graphs(dataset=self.dataset, e_a=True, e_s=True)

        else:
            # Close-set processing
            close_path = f'{self.base_path}/closed_set/'
            os.makedirs(close_path, exist_ok=True)
            
            self.split_and_save_masks(df, close_path)
            self.get_closed_set_test_df(df)
            self.get_closed_set_messages_embeddings()
            self.construct_closed_set_graph(dataset=self.dataset, e_a=True, e_s=True)


    def get_closed_set_test_df(self, df):
        """Get closed set test dataframe"""
        save_path = f'{self.base_path}/closed_set/'
        if not exists(save_path):
            os.makedirs(save_path)
        
        test_set_df_np_path = save_path + 'test_set.npy'
        test_set_label_path = save_path + 'label.npy'
        
        if not exists(test_set_df_np_path):
            # Load test mask from dataset-specific location
            test_mask = torch.load(f'{save_path}/test_mask.pt').cpu().detach().numpy()
            test_indices = list(np.where(test_mask==True)[0])
            
            # Process test data
            test_df = df.iloc[test_indices]
            shuffled_index = np.random.permutation(test_df.index)
            shuffled_df = test_df.reindex(shuffled_index)
            shuffled_df.reset_index(drop=True, inplace=True)
            
            # Save processed data
            test_df_np = shuffled_df.to_numpy()
            labels = [int(label) for label in shuffled_df['event_id'].values]
            
            np.save(test_set_label_path, np.asarray(labels))
            np.save(test_set_df_np_path, test_df_np)

    def get_closed_set_messages_embeddings(self):
        """Get SBERT embeddings for closed set messages"""
        save_path = f'{self.base_path}/closed_set/'
        
        SBERT_embedding_path = f'{save_path}/SBERT_embeddings.pkl'
        if not exists(SBERT_embedding_path):
            test_df_np = np.load(save_path + 'test_set.npy', allow_pickle=True)
            test_df = pd.DataFrame(data=test_df_np, columns=self.columns)
            
            processed_text = [preprocess_sentence(s) for s in test_df['text'].values]
            embeddings = SBERT_embed(processed_text, language=self.language)
            
            with open(SBERT_embedding_path, 'wb') as fp:
                pickle.dump(embeddings, fp)
            print('SBERT embeddings stored.')

    def get_open_set_messages_embeddings(self):
        """Get SBERT embeddings for open set messages"""
        save_path = f'{self.base_path}/open_set/'
        num_blocks = self.dataset.get_num_blocks()  # 从dataset对象获取块数
        
        for block in range(num_blocks):
            print(f'\nProcessing block: {block}')
            SBERT_embedding_path = f'{save_path}{block}/SBERT_embeddings.pkl'
            
            if not exists(SBERT_embedding_path):
                df_np = np.load(f'{save_path}{block}/{block}.npy', allow_pickle=True)
                columns = self.columns + ["original_index", "date"]
                df = pd.DataFrame(data=df_np, columns=columns)
                
                df['processed_text'] = [preprocess_sentence(s) for s in df['text']]
                embeddings = SBERT_embed(df['processed_text'].tolist(), language=self.language)
                
                with open(SBERT_embedding_path, 'wb') as fp:
                    pickle.dump(embeddings, fp)
                print('SBERT embeddings stored.')

    def split_open_set(self, df, root_path):
        """Split data into open set blocks"""
        if not exists(root_path):
            os.makedirs(root_path)
        
        df = df.sort_values(by='created_at').reset_index()
        df['date'] = [d.date() for d in df['created_at']]
        distinct_dates = df.date.unique()

        # Process first week (block 0)
        self._process_block(df, root_path, 0, distinct_dates[:7])

        # Process remaining blocks
        end = len(distinct_dates) - self.dataset.get_end_offset() 
        for i in range(7, end):
            block_num = i - 6
            self._process_block(df, root_path, block_num, [distinct_dates[i]])

    def _process_block(self, df, root_path, block_num, dates):
        """Helper method to process individual blocks"""
        folder = os.path.join(root_path, str(block_num))
        os.makedirs(folder, exist_ok=True)
        
        df_np_path = os.path.join(folder, f'{block_num}.npy')
        label_path = os.path.join(folder, 'label.npy')
        
        if not exists(df_np_path):
            block_df = df.loc[df['date'].isin(dates)]
            shuffled_index = np.random.permutation(block_df.index)
            shuffled_df = block_df.reindex(shuffled_index).reset_index(drop=True)
            
            labels = [int(label) for label in shuffled_df['event_id'].values]
            np.save(label_path, np.asarray(labels))
            np.save(df_np_path, shuffled_df.to_numpy())

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

    def get_best_threshold(self, path):
        """Get best threshold for edge construction"""
        best_threshold_path = path + '/best_threshold.pkl'
        if not os.path.exists(best_threshold_path):
            embeddings_path = path + '/SBERT_embeddings.pkl'
            with open(embeddings_path, 'rb') as f:
                embeddings = pickle.load(f)
            best_threshold = search_threshold(embeddings)
            best_threshold = {'best_thres': best_threshold}
            with open(best_threshold_path, 'wb') as fp:
                pickle.dump(best_threshold, fp)
            print('Best threshold is stored.')

        with open(best_threshold_path, 'rb') as f:
            best_threshold = pickle.load(f)
        print('Best threshold loaded.')
        return best_threshold

    def construct_graph(self, df, embeddings, save_path, e_a=True, e_s=True):
        """Construct graph for given dataframe and embeddings"""
        # Use unified columns
        all_node_features = [[str(u)] + [str(each) for each in um] + [h.lower() for h in hs] + e 
                            for u, um, hs, e in zip(df['user_id'], df['user_mentions'], 
                                                df['hashtags'], df['entities'])]
        
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

    def construct_open_set_graphs(self, dataset, e_a=True, e_s=True, test_with_one_block=False):
        """Construct graphs for open set"""
        dataset_name = dataset.get_dataset_name()
        base_path = '../model/model_saved/hypersed/data_SBERT/'+dataset_name+'/open_set'
        times_save_path = '../model/model_saved/hypersed/graph_times/'+dataset_name+'/'
        times = []

        # Unified columns
        columns = ['tweet_id', 'text', 'event_id', 'words', 'filtered_words',
                'entities', 'user_id', 'created_at', 'urls', 'hashtags', 'user_mentions']

        if test_with_one_block:
            blocks = [dataset.get_num_blocks() - 1]
        else:
            blocks = range(1, dataset.get_num_blocks())

        for block in blocks:
            print('\n\n====================================================')
            print('block: ', block)
            time1 = datetime.now().strftime("%H:%M:%S")
            print('Graph construct starting time:', time1)

            folder = os.path.join(base_path, str(block))

            # Load embeddings and data
            with open(f'{folder}/SBERT_embeddings.pkl', 'rb') as f:
                embeddings = pickle.load(f)
            
            df_np = np.load(f'{folder}/{block}.npy', allow_pickle=True)
            df = pd.DataFrame(data=df_np, columns=columns + ["original_index", "date"])

            # Construct graph
            sparse_adj_matrix, edge_types = self.construct_graph(df, embeddings, folder, e_a, e_s)
            sparse.save_npz(f'{base_path}/{block}/message_graph_{edge_types}.npz', sparse_adj_matrix)

            time2 = datetime.now().strftime("%H:%M:%S")
            print('Graph construct ending time:', time2)
            times.append({'t1': time1, 't2': time2})

        # Save times
        os.makedirs(times_save_path, exist_ok=True)
        with open(f'{times_save_path}hypersed_open.json', "w") as f:
            json.dump(times, f, indent=4)

    def construct_closed_set_graph(self, dataset, e_a=True, e_s=True):
        """Construct graph for closed set"""
        dataset_name = dataset.get_dataset_name()
        base_path = '../model/model_saved/hypersed/data_SBERT/'+dataset_name+'/closed_set'
        times_save_path = '../model/model_saved/hypersed/graph_times/'+dataset_name+'/'

        # Unified columns
        columns = ['tweet_id', 'text', 'event_id', 'words', 'filtered_words',
                'entities', 'user_id', 'created_at', 'urls', 'hashtags', 'user_mentions']

        print('\n\n====================================================')
        time1 = datetime.now().strftime("%H:%M:%S")
        print('Graph construct starting time:', time1)

        # Load embeddings and data
        with open(f'{base_path}/SBERT_embeddings.pkl', 'rb') as f:
            embeddings = pickle.load(f)
        
        df_np = np.load(f'{base_path}/test_set.npy', allow_pickle=True)
        df = pd.DataFrame(data=df_np, columns=columns)

        # Construct graph
        sparse_adj_matrix, edge_types = self.construct_graph(df, embeddings, base_path, e_a, e_s)
        sparse.save_npz(f'{base_path}/message_graph_{edge_types}.npz', sparse_adj_matrix)

        time2 = datetime.now().strftime("%H:%M:%S")
        print('Graph construct ending time:', time2)

        # Save times
        os.makedirs(times_save_path, exist_ok=True)
        with open(f'{times_save_path}hypersed_closed.json', "w") as f:
            json.dump({'t1': time1, 't2': time2}, f, indent=4)


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


def replaceAtUser(text):
    """ Replaces "@user" with "" """
    text = re.sub('@[^\s]+|RT @[^\s]+','',text)
    return text

def removeUnicode(text):
    """ Removes unicode strings like "\u002c" and "x96" """
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r'', text)       
    text = re.sub(r'[^\x00-\x7f]',r'',text)
    return text

def replaceURL(text):
    """ Replaces url address with "url" """
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','url',text)
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
    text = re.sub(':\)|;\)|:-\)|\(-:|:-D|=D|:P|xD|X-p|\^\^|:-*|\^\.\^|\^\-\^|\^\_\^|\,-\)|\)-:|:\'\(|:\(|:-\(|:\S|T\.T|\.\_\.|:<|:-\S|:-<|\*\-\*|:O|=O|=\-O|O\.o|XO|O\_O|:-\@|=/|:/|X\-\(|>\.<|>=\(|D:', '', text)
    return text

def removeNewLines(text):
    text = re.sub('\n', '', text)
    return text

def preprocess_sentence(s):
    return removeNewLines(replaceAtUser(removeEmoticons(replaceMultiQuestionMark(replaceMultiExclamationMark(removeUnicode(replaceURL(s)))))))

def preprocess_french_sentence(s):
    return removeNewLines(replaceAtUser(removeEmoticons(replaceMultiQuestionMark(replaceMultiExclamationMark(replaceURL(s))))))

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


if __name__ == '__main__':
    from dataset.dataloader import Event2012
    dataset = Event2012()
    hypersed = Hypersed(dataset)
    hypersed.preprocess()
    #hypersed.detection()
    #hypersed.evaluate()