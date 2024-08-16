import gc
import random
import numpy as np
import os
import time
from datetime import datetime
from collections import deque
from sklearn import metrics
from scipy.sparse import csr_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import optim
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import pickle
import copy
import argparse
import networkx as nx
import scipy.sparse as sp
from torch_geometric import loader
import en_core_web_lg
import pandas as pd
import networkx as nx
from torch_geometric.data import Data
from torch_geometric import loader
from torch_geometric.nn import GCNConv
from torch.distributions import Categorical, MultivariateNormal
from sklearn.metrics import silhouette_score,calinski_harabasz_score

def currentTime():
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

class args_define:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, default='../model_saved/hcrc/', help="Path for saving files.")
    parser.add_argument("--result_path", type=str, default='../model_saved/hcrc/res.txt', help="Path for saving experimental results. Default is res.txt.")
    parser.add_argument("--task", type=str, default='DRL', help="Name of the task. Supported names are: DRL, random, semi-supervised, traditional. Default is DRL.")
    parser.add_argument("--layers", nargs='?', default='[256]', help="The number of units of each layer of the GNN. Default is [256]")
    parser.add_argument("--N_pred_hid", type=int, default=64, help="The number of hidden units of layer of the predictor. Default is 512")
    parser.add_argument("--G_pred_hid", type=int, default=16, help="The number of hidden units of layer of the predictor. Default is 512")
    parser.add_argument("--eval_freq", type=float, default=5, help="The frequency of model evaluation")
    parser.add_argument("--mad", type=float, default=0.9, help="Moving Average Decay for Teacher Network")
    parser.add_argument("--Glr", type=float, default=0.0000006, help="learning rate")
    parser.add_argument("--Nlr", type=float, default=0.00001, help="learning rate")
    parser.add_argument("--Ges", type=int, default=50, help="Early Stopping Criterion")
    parser.add_argument("--Nes", type=int, default=2000, help="Early Stopping Criterion")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--Gepochs", type=int, default=105)
    parser.add_argument("--Nepochs", type=int, default=100)
    
    args = parser.parse_args()

class HCRC:
    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset

    def detection(self):
        for i in range(22):
            print("************Message Block "+str(i)+" start! ************")
            #Node-level learning
            embedder_N = Node_ModelTrainer(args,i)
            Node_emb,label = embedder_N.get_embedding()
            #Graph-level learning
            embedder_G = Graph_ModelTrainer(args,i)
            Graph_emb,label = embedder_G.get_embedding()
            #combining vectors
            if i==0:
                all_embeddings = np.concatenate((Graph_emb,Node_emb),axis=1)
                all_label = label
            else:
                temp = np.concatenate((Graph_emb,Node_emb),axis=1)
                all_embeddings = np.concatenate((all_embeddings,temp),axis=0)
                all_label = all_label+label
            all_embeddings = torch.tensor(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, dim=-1, p=2).detach().cpu().numpy()
                
            if i == 0:
                pred_y = evaluate_fun(all_embeddings,label,i,None,args.result_path,args.task)
                all_pred_y = pred_y
            else:
                pred_y = evaluate_fun(all_embeddings,label,i,all_pred_y,args.result_path,args.task)
                all_pred_y = all_pred_y + pred_y
            print("************Message Block "+str(i)+" end! ************\n\n")

        predictions = all_pred_y
        ground_truths = all_label

        return predictions, ground_truths

    def evaluate(self, predictions, ground_truths):
        print("************Evaluation start! ************")
        ars = metrics.adjusted_rand_score(ground_truths, predictions)

        # Calculate Adjusted Mutual Information (AMI)
        ami = metrics.adjusted_mutual_info_score(ground_truths, predictions)

        # Calculate Normalized Mutual Information (NMI)
        nmi = metrics.normalized_mutual_info_score(ground_truths, predictions)
        
        print(f"Model Adjusted Rand Index (ARI): {ars}")
        print(f"Model Adjusted Mutual Information (AMI): {ami}")
        print(f"Model Normalized Mutual Information (NMI): {nmi}")
        return ars, ami, nmi

class EMA:  #Exponential Moving Average
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def get_task(strs):
    tasks = ["DRL","random","semi-supervised","traditional"]
    if len(strs) == 1:
        return "DRL"
    if ("--task" in strs) and len(strs) == 2:
        return "DRL"
    if ("--task" not in strs) or len(strs)!=3:
        return False
    elif strs[-1] not in tasks:
        return False
    else:
        return strs[-1]

def init_weights(m):  #Model parameter initialization
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
def sim(z1, z2):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def semi_loss(z1, z2):
    f = lambda x: torch.exp(x / 0.05)
    refl_sim = f(sim(z1, z1))
    between_sim = f(sim(z1, z2))

    return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

def get_loss(h1, h2):
    l1 = semi_loss(h1, h2)
    l2 = semi_loss(h2, h1)

    ret = (l1 + l2) * 0.5
    ret = ret.mean()

    return ret

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def set_requires_grad(model, val):
    #set require_grad
    for p in model.parameters():
        p.requires_grad = val

def enumerateConfig(args):
    args_names = []
    args_vals = []
    for arg in vars(args):
        args_names.append(arg)
        args_vals.append(getattr(args, arg))

    return args_names, args_vals

def config2string(args):
    args_names, args_vals = enumerateConfig(args)
    st = ''
    for name, val in zip(args_names, args_vals):
        if val == False:
            continue
        if name not in ['device','root','epochs','isAnneal','dropout','warmup_step','clus_num_iters']:
            st_ = "{}_{}_".format(name, val)
            st += st_

    return st[:-1]

def printConfig(args): 
    args_names, args_vals = enumerateConfig(args)
    print(args_names)
    print(args_vals)


class embedder:
    def __init__(self, args):
        self.args = args
        self.hidden_layers = eval(args.layers)

class Encoder(nn.Module):

    def __init__(self, layer_config):
        super().__init__()
        self.stacked_gnn = nn.ModuleList(
            [GCNConv(layer_config[i - 1], layer_config[i]) for i in range(1, len(layer_config))])
        self.stacked_bns = nn.ModuleList(
            [nn.BatchNorm1d(layer_config[i], momentum=0.01) for i in range(1, len(layer_config))])
        self.stacked_prelus = nn.ModuleList([nn.PReLU() for _ in range(1, len(layer_config))])

    def forward(self, x, edge_index):
        for i, gnn in enumerate(self.stacked_gnn):
            x = gnn(x, edge_index, edge_weight=None)
            x = self.stacked_bns[i](x)
            x = self.stacked_prelus[i](x)

        return x

M =[20254,28976,30467,32302,34312,36146,37422,42700,44260,45623,46719,
    47951,51188,53160,56116,58665,59575,62251,64138,65537,66430,68840]

def DRL_cluster(all_embeddings,block_num,pred_label):
    para = 0.1
    if block_num == 0:
        print("Evaluating initial message block...")
        start_time = time.time()
        sp = SinglePass(0.87, all_embeddings, 0, pred_label, M[0], None, para, 0, sim=False)
        end_time = time.time()
        run_time = end_time - start_time
        print("Done! " + "It takes "+str(int(run_time))+" seconds.\n")
    else:
        print("Using DRL-Single-Pass to learn threshold...")
        global_step = 0
        agent = PPO([5], 1, continuous=True)
        sp_sim = SinglePass(0.6, all_embeddings, 1, pred_label, M[block_num] - M[block_num - 1], agent, para, M[block_num-1]-2000, sim=True)
        
        global_step = sp_sim.global_step
        sp = SinglePass(0.6, all_embeddings, 1, pred_label, M[block_num] - M[block_num - 1], agent, para, M[block_num-1]-2000, sim=False)
    
    return sp.cluster_result,sp.sim_threshold
    
def random_cluster(all_embeddings,block_num,pred_label):
    threshold = random.uniform(0.6,0.8)
    if block_num == 0:
        print("Evaluating initial message block...")
        start_time = time.time()
        sp = SinglePass(0.87, all_embeddings, 0, pred_label, M[0], None, 0, 0, sim=False)
        end_time = time.time()
        run_time = end_time - start_time
        print("Done! " + "It takes "+str(int(run_time))+" seconds.\n")
        threshold = 0.87
    else:
        print("Evaluating message block...")
        start_time = time.time()
        sp = SinglePass(threshold, all_embeddings, 2, pred_label, M[block_num] - M[block_num - 1], None, 0, 0, sim=False)
        end_time = time.time()
        run_time = end_time - start_time
        print("Done! " + "It takes "+str(int(run_time))+" seconds.\n")
    return sp.cluster_result,threshold
    
def semi_cluster(all_embeddings,label,block_num,pred_label):
    if block_num == 0:
        print("Evaluating initial message block...")
        start_time = time.time()
        sp = SinglePass(0.87, all_embeddings, 0, pred_label, M[0], None, 0, 0, sim=False)
        end_time = time.time()
        run_time = end_time - start_time
        print("Done! " + "It takes "+str(int(run_time))+" seconds.\n")
        threshold = 0.87
    else:
        print("Evaluating message block...")
        start_time = time.time()
        embeddings = all_embeddings.tolist()
        size = M[block_num] - M[block_num - 1]
        embeddings = embeddings[0:len(embeddings)-int(size*0.9)]
        pre_label = pred_label[0:len(embeddings)]
        
        size = len(embeddings) - M[block_num - 1]
        embeddings = np.array(embeddings)
        thresholds = [0.6,0.65,0.7,0.75,0.8]
        s1s = []
        for t in thresholds:
            sp = SinglePass(t, embeddings, 2, pre_label, size, None, 0, 0, sim=False)
            true_label = label[0:len(sp.cluster_result)]
            s1 = metrics.normalized_mutual_info_score(true_label, sp.cluster_result, average_method='arithmetic')
            s1s.append(s1)
        index = s1s.index(max(s1s))
        sp = SinglePass(thresholds[index], all_embeddings, 2, pred_label, M[block_num] - M[block_num - 1], None, 0, 0, sim=False)
        end_time = time.time()
        run_time = end_time - start_time
        print("Done! " + "It takes "+str(int(run_time))+" seconds.\n")
        threshold = thresholds[index]
    return sp.cluster_result,threshold
        
def NMI_cluster(all_embeddings,label,block_num,pred_label):
    if block_num == 0:
        print("Evaluating initial message block...")
        start_time = time.time()
        sp = SinglePass(0.87, all_embeddings, 0, pred_label, M[0], None, 0, 0, sim=False)
        end_time = time.time()
        run_time = end_time - start_time
        print("Done! " + "It takes "+str(int(run_time))+" seconds.\n")
        threshold = 0.87
    else:
        print("Evaluating message block...")
        start_time = time.time()
        thresholds = [0.6,0.65,0.7,0.75,0.8]
        s1s = []
        for t in thresholds:
            sp = SinglePass(t, all_embeddings, 2, pred_label, M[block_num] - M[block_num - 1], None, 0, 0, sim=False)
            s1 = metrics.normalized_mutual_info_score(label, sp.cluster_result, average_method='arithmetic')
            s1s.append(s1)
        index = s1s.index(max(s1s))
        sp = SinglePass(thresholds[index], all_embeddings, 2, pred_label, M[block_num] - M[block_num - 1], None, 0, 0, sim=False)
        end_time = time.time()
        run_time = end_time - start_time
        print("Done! " + "It takes "+str(int(run_time))+" seconds.\n")
        threshold = thresholds[index]
    return sp.cluster_result,threshold 
        
def evaluate_fun(all_embeddings,label,block_num,pred_label,result_path,task):
    if task == "DRL":
        y_pred,threshold = DRL_cluster(all_embeddings,block_num,pred_label)
    elif task == "random":
        y_pred,threshold = random_cluster(all_embeddings,block_num,pred_label)
    elif task == "semi-supervised":
        y_pred,threshold = semi_cluster(all_embeddings,label,block_num,pred_label)
    elif task == "traditional":
        y_pred,threshold = NMI_cluster(all_embeddings,label,block_num,pred_label)
    
    #NMI
    s1 = metrics.normalized_mutual_info_score(label, y_pred, average_method='arithmetic')
    #AMI
    s2 = metrics.adjusted_mutual_info_score(label, y_pred, average_method='arithmetic')
    #ARI
    s3 = metrics.adjusted_rand_score(label, y_pred)
    
    print('** Theta:{:.2f} **\n'.format(threshold))
    print('** NMI: {:.2f} **\n'.format(s1))
    print('** AMI: {:.2f} **\n'.format(s2))
    print('** ARI: {:.2f} **\n'.format(s3))
    result = '\nmessage_block_'+str(block_num)+'\nthreshold: {:.2f} '.format(threshold)+'\n** NMI: {:.2f} **\n'.format(s1) + '** AMI: {:.2f} **\n'.format(s2) + '** ARI: {:.2f} **\n'.format(s3)

    if not os.path.exists(result_path) :
        pass
    else:
        with open(result_path,encoding='utf-8') as file:
            content=file.read()
        result = content.rstrip() + result
    file = open(result_path, mode='w')
    file.write(result)
    file.close()
    return y_pred

class SinglePass:
    def __init__(self, sim_threshold, data, flag, label, size, agent, para, sim_init, sim=False, global_step=0):
        self.device = torch.device('cuda:0')
        self.text_vec = None  #
        self.topic_serial = None
        self.topic_cnt = 0
        self.sim_threshold = sim_threshold
        
        self.done_data = data[0:data.shape[0] - size]
        self.new_data = data[data.shape[0] - size:]
        self.done_label = label
        
        if flag == 0 or flag == 2:
            self.cluster_result = self.run_cluster(flag, size)
        else:
            self.agent = agent
            self.scheme = ["state", "action", "reward", "done", "log_prob"]
            self.global_step = global_step
            self.sim = sim
            if self.sim:
                start_time = time.time()
                self.cluster_result = self.run_cluster_sim(flag, size, para, sim_init, sim, data)  
                end_time = time.time()
                self.time = end_time - start_time
                print("Creating Environment Done! " + "It takes "+str(int(self.time))+" seconds.")
            else:
                start_time = time.time()
                self.pseudo_labels = self.run_cluster_init(0.6, size)
                if flag == 1:
                    self.text_vec = self.done_data
                    self.topic_serial = copy.deepcopy(self.done_label)
                    self.topic_cnt = max(self.topic_serial)
                state = self.get_state(sim, sim_init, data)
                action, action_log_prob = self.agent.select_action(state)
                # action projection
                sim_threshold = torch.clamp(action, -1, 1).detach()
                sim_threshold += 7
                sim_threshold /=10
                self.sim_threshold = sim_threshold.item()
                end_time = time.time()
                self.time = end_time - start_time
                print("Getting Threshold Done! " + "It takes "+str(int(self.time))+" seconds. ")
                print("Threshold is "+str(self.sim_threshold)+".\n")
                print("Evaluating message block...")
                start_time = time.time()
                self.cluster_result = self.run_cluster(flag, size)  # clustering
                end_time = time.time()
                self.time = end_time - start_time
                print("Done! " + "It takes "+str(int(self.time))+" seconds.\n")

    def clustering(self, sen_vec):
        if self.topic_cnt == 0:
            self.text_vec = sen_vec
            self.topic_cnt += 1
            self.topic_serial = [self.topic_cnt]
        else:
            sim_vec = np.dot(sen_vec, self.text_vec.T)
            max_value = np.max(sim_vec)

            topic_ser = self.topic_serial[np.argmax(sim_vec)]
            self.text_vec = np.vstack([self.text_vec, sen_vec])

            if max_value >= self.sim_threshold:
                self.topic_serial.append(topic_ser)
            else:
                self.topic_cnt += 1
                self.topic_serial.append(self.topic_cnt)
    
    def clustering_init(self, t, sen_vec):
        if self.topic_cnt_init == 0:
            self.text_vec_init = sen_vec
            self.topic_cnt_init += 1
            self.topic_serial_init = [self.topic_cnt_init]
        else:
            sim_vec = np.dot(sen_vec, self.text_vec_init.T)
            max_value = np.max(sim_vec)

            topic_ser = self.topic_serial_init[np.argmax(sim_vec)]
            self.text_vec_init = np.vstack([self.text_vec_init, sen_vec])

            if max_value >= t:
                self.topic_serial_init.append(topic_ser)
            else:
                self.topic_cnt_init += 1
                self.topic_serial_init.append(self.topic_cnt_init)
    
    def run_cluster_init(self, t, size):
        self.text_vec_init = []
        self.topic_serial_init = []
        self.topic_cnt_init = 0
        for vec in self.new_data:
            self.clustering_init(t,vec)
        return self.topic_serial_init
    
    def run_cluster_sim(self, flag, size, para, sim_init, sim, data):
        self.text_vec = []
        self.topic_serial = []
        self.topic_cnt = 0
        if flag == 1:
            self.text_vec = self.done_data
            self.topic_serial = copy.deepcopy(self.done_label)
            self.topic_cnt = max(self.topic_serial)
        for i, vec in enumerate(self.new_data):
            self.global_step += 1
            if i > 200:
                break
            if i > self.new_data.shape[0] * para:
                break
            state = self.get_state(sim, sim_init, data)
            
            action, action_log_prob = self.agent.select_action(state)
            self.sim_threshold = action.item()
            self.clustering(vec)
            
            reward = self.get_reward(sim_init, data)
            done = False
            transition = make_transition(self.scheme, state, action, reward, done, action_log_prob)
            self.agent.add_buffer(transition)
            if self.global_step % 200==0:
                self.agent.learn()

        return self.topic_serial[len(self.topic_serial) - size:]

    def run_cluster(self, flag, size):
        self.text_vec = []
        self.topic_serial = []
        self.topic_cnt = 0
        if flag == 1 or flag == 2:
            self.text_vec = self.done_data
            self.topic_serial = copy.deepcopy(self.done_label)
            self.topic_cnt = max(self.topic_serial)
        for i, vec in enumerate(self.new_data):
            self.clustering(vec)
        return self.topic_serial[len(self.topic_serial) - size:]

    def get_center(self,label,data):
        centers = []
        indexs_per_cluster = []
        label_u = list(set(label))
        for i in range(len(label_u)):
            indexs = [False] * data.shape[0]
            tmp_indexs_text = []
            for j in range(len(indexs)):
                if label[j] == label_u[i]:
                    indexs[j] = True
                    tmp_indexs_text.append(j)
            center = np.mean(data[indexs], 0).tolist()
            centers.append(center)
            indexs_per_cluster.append(tmp_indexs_text)
        return centers,indexs_per_cluster

    def get_info_cluster(self,text_vec,indexs_per_cluster):  # Get detailed clustering results
        res = []
        for i in range(len(indexs_per_cluster)):
            tmp_vec = []
            for j in range(len(indexs_per_cluster[i])):
                tmp_vec.append(text_vec[indexs_per_cluster[i][j]])
            tmp_vec = np.array(tmp_vec)
            res.append(tmp_vec)
        return res

    def get_state(self, sim, sim_init, data):  # get state of RL
        state = []
        if sim:
            data = data[sim_init:len(self.topic_serial)]
            topic_serial = self.topic_serial[sim_init:]
        else:
            data = self.new_data
            topic_serial = self.pseudo_labels
        centers,indexs_per_cluster = self.get_center(topic_serial, data)
        
        centers = np.array(centers)
        neighbor_dists = np.dot(centers, centers.T)
        
        neighbor_dists = np.nan_to_num(neighbor_dists, 0.0001)
        # the minimum neighbor distance
        state.append(neighbor_dists.min())
        # the average separation distance
        state.append((neighbor_dists.mean() * max(topic_serial) - 1) / max(topic_serial))
        info_of_cluster = self.get_info_cluster(data,indexs_per_cluster)

        coh_dists = 0
        for cluster in info_of_cluster:
            if cluster.shape[0] == 1:
                continue
            else:
                sums = cluster.shape[0] * (cluster.shape[0] - 1) / 2
            tmp_vec = np.array(cluster)
            cohdist = np.dot(tmp_vec, tmp_vec.T)
            if cohdist.max() > coh_dists:
                coh_dists = cohdist.max()
#         Dunn index
        state.append(neighbor_dists.min()/coh_dists)
    
        #Sum of intra-group error squares
        SSE = 0
        SSEE = 0
        for i in range(len(indexs_per_cluster)):
            sumtmp = 0
            for j in range(len(indexs_per_cluster[i])):
                tmp = np.dot(data[indexs_per_cluster[i][j]].T,centers[i])
                SSE = SSE + (tmp)**2
                sumtmp = sumtmp + (tmp)**2
            SSEE = SSEE + sumtmp/len(indexs_per_cluster[i])
#         state.append(SSE)

        # Sum of squared errors between groups
        SSR = 0
        SSRR = 0
        for i in range(len(centers)):
            SSR = SSR + np.dot(centers[i].T,centers.mean(axis=0))
            SSRR = SSRR + np.dot(centers[i].T,centers.mean(axis=0))**2
        SSRR = SSRR / max(topic_serial)
#         state.append(SSR)
        #the average cohesion distance
        coh_dists = 0
        for cluster in info_of_cluster:
            if cluster.shape[0] == 1:
                continue
            else:
                sums = cluster.shape[0] * (cluster.shape[0] - 1) / 2
            tmp_vec = np.array(cluster)
            cohdist = np.dot(tmp_vec, tmp_vec.T)
            cohdist = np.maximum(cohdist, -cohdist)
            coh_dists = coh_dists + (cohdist.sum() - cluster.shape[0]) / (2 * sums + 0.0001)
        state.append(coh_dists / max(topic_serial))

        state.append(silhouette_score(data, topic_serial, metric='euclidean'))
        return np.array(state)

    def get_reward(self, sim_init, data):  # get reward of RL

        data = data[sim_init:len(self.topic_serial)]
        topic_serial = self.topic_serial[sim_init:]
        
        return calinski_harabasz_score(data, topic_serial)

class Node_ModelTrainer(embedder):

    def __init__(self, args,block_num):
        embedder.__init__(self, args)
        self._args = args
        self.block_num = block_num
        self._init()

    def _init(self):
        block_num = self.block_num
        args = self._args
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self._device)
        
        
        if block_num == 0 or block_num == 1:
            args.Nes = 200
        else:
            args.Nes = 2000
        self.true_label = []
        #load data
        self._loader,self.true_label = get_Node_Dataset(block_num)

        layers = [302] + self.hidden_layers
        self._model = NodeLevel(layers, args)

        self._optimizer = optim.AdamW(params=self._model.parameters(), lr=args.Nlr, weight_decay=1e-5)
        self.train()
        self.all_embeddings = F.normalize(self.all_embeddings, dim=-1, p=2).detach().cpu().numpy()
    
    def get_embedding(self):
        return self.all_embeddings,self.true_label
    
    def train(self):

        loss_t = 1e10
        cnt_wait = 0
        # Start Model Training
        print("----Node-Level Training Start! ----\n")
        for epoch in range(self._args.Nepochs):
            losses = []
            embs = []
            for batch in self._loader:

                emb,loss = self._model(batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._model.update_moving_average()
                losses.append(loss.item())
                embs = embs +emb

            st = '[{}][Epoch {}/{}] Loss: {:.4f}'.format(currentTime(), 
                                                         epoch, self._args.Nepochs, np.mean(np.array(losses)))
            print(st)
            #Early Stopping Criterion
            if  np.mean(np.array(losses)) < loss_t:
                loss_t = np.mean(np.array(losses))
            else:
                cnt_wait = cnt_wait + 1
            if cnt_wait > self._args.Nes:
                print("Early Stopping Criterion")
                break

        self.all_embeddings  = torch.tensor(embs)
        print("\n----Node-Level Training Done! ----\n")

class NodeLevel(nn.Module):
    def __init__(self, layer_config, args):
        super().__init__()
        self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        #encoder
        self.student_encoder = Encoder(layer_config=layer_config)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.student_encoder = self.student_encoder.to(self._device)
        self.teacher_encoder = self.teacher_encoder.to(self._device)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(args.mad, args.Nepochs)

        rep_dim = layer_config[-1]

        #projection head
        self.student_projector = nn.Sequential(nn.Linear(rep_dim, args.N_pred_hid), nn.BatchNorm1d(args.N_pred_hid),
                                               nn.PReLU(), nn.Linear(args.N_pred_hid, rep_dim))
     
        self.student_projector = self.student_projector.to(self._device)

        self.student_projector.apply(init_weights)
        self.teacher_projector = copy.deepcopy(self.student_projector)
        set_requires_grad(self.teacher_projector, False)
        
    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)
        update_moving_average(self.teacher_ema_updater, self.teacher_projector, self.student_projector)

    def forward(self, batch):

        student = self.student_encoder(batch.x1.to(torch.float32).to(self._device),batch.edge_index1.to(self._device))

        h1 = self.student_projector(student)
        
        with torch.no_grad(): #stop gradient
            teacher = self.teacher_encoder(batch.x2.to(torch.float32).to(self._device),
                                           batch.edge_index2.to(self._device))
        with torch.no_grad(): #stop gradient
            h2 = self.teacher_projector(teacher)

        emb = self.student_encoder(batch.x.to(torch.float32).to(self._device),batch.edge_index.to(self._device))
   
        emb = emb.detach().cpu().numpy().tolist()

        loss = get_loss(h1,h2)              
          
        return emb,loss

class Graph_ModelTrainer(embedder):

    def __init__(self, args,block_num):
        self.block_num = block_num
        embedder.__init__(self, args)
        self._args = args
        self._init()
    
    def _init(self):
        args = self._args
        block_num = self.block_num
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self._device)
        if torch.cuda.is_available():
            print("using cuda")

        layers = [300] + self.hidden_layers
        #load data
        self._model = GraphLevel(layers, args).to(self._device)
        self._model.to(self._device)
        self._optimizer = optim.AdamW(params=self._model.parameters(), lr=args.Glr, weight_decay=1e-5)

        self._loader,self.true_label = get_Graph_Dataset(block_num)
        self.train()
        self.all_embeddings = F.normalize(self.all_embeddings, dim=-1, p=2).detach().cpu().numpy()


    def get_embedding(self):
        return self.all_embeddings,self.true_label
            
    def train(self):
        h_loss = 1e10
        cnt_wait = 0
        # Start Model Training
        print("----Graph-Level Training Start! ----\n")
        for epoch in range(self._args.Gepochs):
            losses = []
            embs = []
            for batch in self._loader:
                batch = batch.to(self._device)
                emb,loss = self._model(batch)
                self._optimizer.zero_grad() 
                loss.backward()
                self._optimizer.step()
                self._model.update_moving_average()
                losses.append(loss.item())
                embs = embs + emb
            st = '[{}][Epoch {}/{}] Loss: {:.4f}'.format(currentTime(), 
                                                     epoch,self._args.Gepochs, np.mean(np.array(losses)))
            print(st)

            #Early Stopping Criterion
            if np.mean(np.array(losses)) < h_loss:
                h_loss = np.mean(np.array(losses)) 
            elif np.mean(np.array(losses)) > h_loss:
                cnt_wait = cnt_wait + 1
            if cnt_wait > 5:
                break
        self.all_embeddings = torch.tensor(embs)
        print("\n----Graph-Level Training Done! ----")
        
class GraphLevel(nn.Module):
    def __init__(self, layer_config, args):

        self.args = args
        super().__init__()
        self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        #encoder
        self.student_encoder = Encoder(layer_config)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        self.student_encoder.to(self._device)
        self.teacher_encoder.to(self._device)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(args.mad, args.Gepochs)

        rep_dim = layer_config[-1]
        #projection head
        self.student_projector = nn.Sequential(nn.Linear(rep_dim, args.G_pred_hid), nn.BatchNorm1d(args.G_pred_hid),
                                               nn.PReLU(), nn.Linear(args.G_pred_hid, rep_dim))
        self.student_projector.to(self._device)
        self.student_projector.apply(init_weights)
        self.teacher_projector = copy.deepcopy(self.student_projector)
        set_requires_grad(self.teacher_projector, False)
        #pooling
        self.pool = GlobalAttention(gate_nn=nn.Sequential(
                nn.Linear(rep_dim, rep_dim), nn.BatchNorm1d(rep_dim), nn.ReLU(), nn.Linear(rep_dim, 1))) 
  
    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None

    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)
        update_moving_average(self.teacher_ema_updater, self.teacher_projector, self.student_projector)

    def forward(self,batch):

        student = self.student_encoder(batch.x1.to(self._device),batch.edge_index1.to(self._device))

        h1 = self.pool(student,batch.batch)
        h1 = self.student_projector(h1)

        with torch.no_grad(): #stop gradient
            teacher = self.teacher_encoder(batch.x2.to(self._device),batch.edge_index2.to(self._device))
        h2 = self.pool(teacher,batch.batch)
        with torch.no_grad(): #stop gradient
            h2 = self.teacher_projector(h2)

        emb = self.student_encoder(batch.x.to(self._device),batch.edge_index.to(self._device))
        emb = self.pool(emb,batch.batch)
        emb = emb.detach().cpu().numpy().tolist()
        loss = get_loss(h1,h2)
        res_emb = emb

        return res_emb,loss

def make_transition(trans, *items):
    transition = {}
    for key, item in zip(trans, items):
        if isinstance(item, list):
            item = torch.stack(item)
            transition[key] = item
        elif isinstance(item, np.ndarray):
            item = torch.from_numpy(item)
            transition[key] = item
        elif isinstance(item, torch.Tensor):
            transition[key] = item
        else:
            transition[key] = torch.Tensor([item])

    return transition

def make_batch(state, action, old_log_prob, advantage, old_value, learn_size, batch_size, use_cuda):
    batch = []
    total_indices = torch.randperm(learn_size)
    for i in range(learn_size // batch_size):
        indices = total_indices[batch_size * i: batch_size * (i + 1)]
        mini_state = torch.Tensor([])
        mini_action = torch.Tensor([])
        mini_old_log_prob = torch.Tensor([])
        mini_advantage = torch.Tensor([])
        mini_old_value = torch.Tensor([])
        if use_cuda:
            mini_state = mini_state.cuda()
            mini_action = mini_action.cuda()
            mini_old_log_prob = mini_old_log_prob.cuda()
            mini_advantage = mini_advantage.cuda()
            mini_old_value = mini_old_value.cuda()
        for ind in indices:
            mini_state = torch.cat((mini_state, state[ind].unsqueeze(0)), dim=0)
            mini_action = torch.cat((mini_action, action[ind].unsqueeze(0)), dim=0)
            mini_old_log_prob = torch.cat((mini_old_log_prob, old_log_prob[ind].unsqueeze(0)), dim=0)
            mini_advantage = torch.cat((mini_advantage, advantage[ind].unsqueeze(0)), dim=0)
            mini_old_value = torch.cat((mini_old_value, old_value[ind].unsqueeze(0)), dim=0)
        batch.append([mini_state, mini_action, mini_old_log_prob, mini_advantage, mini_old_value])
    return batch

def calculate_nature_cnn_out_dim(height, weight):
    size_h = np.floor((height - 8) / 4) + 1
    size_h = np.floor((size_h - 4) / 2) + 1
    size_h = np.floor((size_h - 3) / 1) + 1
    size_w = np.floor((weight - 8) / 4) + 1
    size_w = np.floor((size_w - 4) / 2) + 1
    size_w = np.floor((size_w - 3) / 1) + 1
    return size_h, size_w

class DQN_Config:
    def __init__(self, input_type, input_size=None):
        self.max_buffer = 100000
        self.update_freq = 200
        self.use_cuda = True
        self.trans = ["state", "action", "reward", "next_state", "done"]
        self.lr = 0.001
        self.tau = 0.005
        self.gamma = 0.99
        self.batch_size = 128
        self.max_grad_norm = 1
        self.epsilon_init = 1
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.q_layer = [256, 256]
        if input_type == "vector":
            self.encoder = "mlp"
            self.encoder_layer = [512, 256]
            self.feature_dim = 256
        elif input_type == "image":
            self.encoder = "cnn"
            self.encoder_layer = [[input_size[0], 32, 8, 4],
                                  [32, 64, 4, 2],
                                  [64, 64, 3, 1]]
            size_h, size_w = calculate_nature_cnn_out_dim(input_size[1], input_size[2])
            self.feature_dim = [int(64 * size_h * size_w), 256]

class DQN:
    def __init__(self, state_dim, action_dim, input_type="vector", args=None):
        if args is None:
            self.args = DQN_Config(input_type, state_dim)
        self.action_dim = action_dim
        self.buffer = BaseBuffer(self.args.trans, self.args.max_buffer)
        self.policy_net = QNet(state_dim[0], action_dim, self.args.q_layer, self.args.encoder, self.args.encoder_layer,
                               self.args.feature_dim)
        self.target_net = QNet(state_dim[0], action_dim, self.args.q_layer, self.args.encoder, self.args.encoder_layer,
                               self.args.feature_dim)
        if self.args.use_cuda:
            self.policy_net = self.policy_net.cuda()
            self.target_net = self.target_net.cuda()
        self.update_network()
        self.policy_net.eval()
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.args.lr)

        self.epsilon = self.args.epsilon_init

    def select_action(self, state, epsilon=None):
        if epsilon is None:
            epsilon = self.epsilon
        if random.random() > epsilon:
            state = torch.Tensor(state)
            if self.args.use_cuda:
                state = state.cuda()
            q_value = self.policy_net(state)
            return torch.argmax(q_value).cpu().unsqueeze(0).detach()
        else:
            return torch.Tensor([random.choice(np.arange(self.action_dim))]).type(torch.int64).detach()

    def add_buffer(self, transition):
        self.buffer.add(transition)

    def epsilon_decay(self):
        self.epsilon = max(self.epsilon * self.args.epsilon_decay, self.args.epsilon_min)

    def update_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, model_path):
        torch.save(self.policy_net.state_dict(), model_path)

    def learn(self, step):
        self.policy_net.train()
        data = self.buffer.sample(self.args.batch_size)
        states = torch.stack(data["state"])  # [batch_size, state_dim]
        actions = torch.stack(data["action"])  # [batch_size, 1]
        rewards = torch.stack(data["reward"])  # [batch_size, 1]
        next_states = torch.stack(data["next_state"])  # [batch_size, state_dim]
        dones = torch.stack(data["done"])  # [batch_size, 1]
        # print("shape check", states.shape, actions.shape, rewards.shape, next_states.shape, dones.shape)
        if self.args.use_cuda:
            states = states.cuda()
            actions = actions.cuda()
            rewards = rewards.cuda()
            next_states = next_states.cuda()
            dones = dones.cuda()
        actions = actions.type(torch.int64)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # get q-values for all actions in current states
        predicted_q = self.policy_net(states)  # [batch_size, action_dim]
        # select q-values for chosen actions
        predicted_q_actions = torch.gather(predicted_q, -1, actions)  # [batch_size, 1]
        # compute q-values for all actions in next states
        predicted_next_q = self.target_net(next_states)  # [batch_size, action_dim]
        # compute V*(next_states) using predicted next q-values
        next_state_values, indexes = torch.max(predicted_next_q, dim=-1)  # [batch_size]
        next_state_values = next_state_values.unsqueeze(-1)  # [batch_size, 1]
        # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
        target_q_actions = rewards + self.args.gamma * next_state_values * (1 - dones)  # [batch_size, 1]
        loss = nn.SmoothL1Loss()(predicted_q_actions, target_q_actions.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.policy_net.eval()
        self.epsilon_decay()

        if step % self.args.update_freq == 0:
            print("update target network")
            self.update_network()
        return loss.item()

class PPO_Config:
    def __init__(self, input_type, input_size=None):
        self.max_buffer = 2048
        self.trainable_std = False
        self.use_cuda = True
        self.trans = ["state", "action", "reward", "done", "log_prob"]
        self.lr = 0.0003
        self.gamma = 0.99
        self.lambda_ = 0.95

        self.train_epoch = 80
        self.clip_ratio = 0.2
        self.critic_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5

        self.action_std_init = 0.6
        self.action_std_decay_rate = 0.05
        self.action_std_min = 0.1
        self.action_std_update_freq = 100

        self.actor_layer = [32, 32]
        self.critic_layer = [32, 32]
        if input_type == "vector":
            self.encoder = "mlp"
            self.encoder_layer = [64, 64]
            self.feature_dim = 32
        elif input_type == "image":
            self.encoder = "cnn"
            self.encoder_layer = [[input_size[0], 32, 8, 4],
                                  [32, 64, 4, 2],
                                  [64, 64, 3, 1]]
            size_h, size_w = calculate_nature_cnn_out_dim(input_size[1], input_size[2])
            self.feature_dim = [int(64 * size_h * size_w), 256]

class PPO:
    def __init__(self, state_dim, action_dim, continuous=True, input_type="vector", args=None):
        if args is None:
            self.args = PPO_Config(input_type, state_dim)
        self.buffer = BaseBuffer(self.args.trans, self.args.max_buffer)
        self.model = ActorCritic(state_dim[0], action_dim, self.args.actor_layer, self.args.critic_layer,
                                 self.args.encoder, self.args.encoder_layer, self.args.feature_dim, continuous,
                                 self.args.action_std_init)
        if self.args.use_cuda:
            self.model = self.model.cuda()
        self.model.eval()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

    def select_action(self, state):
        state = torch.Tensor(state).float()
        if self.args.use_cuda:
            state = state.cuda()
        action, action_log_prob = self.model.act(state)
        return action.detach().cpu(), action_log_prob.detach().cpu()

    def add_buffer(self, transition):
        self.buffer.add(transition)

    def save_model(self, model_path):
        torch.save(self.model.state_dict(), model_path)

    def learn(self):
        self.model.train()
        data, size = self.buffer.get_data_buffer()
        states = data["state"]
        actions = data["action"]
        rewards = data["reward"]
        dones = data["done"]
        old_log_probs = data["log_prob"]

        # Monte Carlo estimate of returns
        returns = []
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.args.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        # Normalizing the rewards
        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        # trans into tensor and send to cpu/gpu
        states = torch.stack(states).float()
        actions = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs)
        if self.args.use_cuda:
            states = states.cuda()  # [batch_size, state_dim]
            actions = actions.cuda()  # [batch_size]
            old_log_probs = old_log_probs.cuda()  # [batch_size]
            returns = returns.cuda()  # [batch_size]

        loss_list = []
        for e in range(self.args.train_epoch):
            # Evaluating old actions and values
            log_probs, values, dist_entropy = self.model.evaluate_AC(states, actions)
            # print("shape check", log_probs.shape, values.shape, dist_entropy.shape)
            values = values.squeeze(-1)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(log_probs - old_log_probs.detach())

            # Finding Surrogate Loss
            advantages = returns - values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.args.clip_ratio, 1 + self.args.clip_ratio) * advantages

            # final loss of clipped objective PPO
            actor_loss = -torch.min(surr1, surr2)
            critic_loss = nn.MSELoss()(values, returns)
            entropy_bonus = -dist_entropy
            loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_bonus
            loss = loss.mean()

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_list.append(loss.item())
        self.model.eval()
        self.buffer.clear()
        return np.mean(loss_list)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, actor_layer, critic_layer, encoder=None, encoder_layer=None,
                 feature_dim=None, continuous=False, std_init=0.6):
        super(ActorCritic, self).__init__()
        self.continuous = continuous
        if continuous:
            self.action_var = torch.full((action_dim,), std_init * std_init)

        self.encoder = encoder
        if encoder is None:
            input_dim = state_dim
        elif encoder == "mlp":
            self.encoder = MLPEncoder(state_dim, encoder_layer, feature_dim)
            input_dim = self.encoder.get_dim()
        elif encoder == "cnn":
            self.encoder = CNNEncoder(encoder_layer, feature_dim)
            input_dim = self.encoder.get_dim()
        else:
            raise NotImplementedError

        if self.continuous:
            layers = [nn.Linear(input_dim, actor_layer[0]),
                      nn.ReLU(inplace=True)]
            for i in range(len(actor_layer) - 1):
                layers.append(nn.Linear(actor_layer[i], actor_layer[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(actor_layer[-1], action_dim))
            self.actor = nn.Sequential(*layers)

        else:
            layers = [nn.Linear(input_dim, actor_layer[0]),
                      nn.ReLU(inplace=True)]
            for i in range(len(actor_layer) - 1):
                layers.append(nn.Linear(actor_layer[i], actor_layer[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(actor_layer[-1], action_dim))
            layers.append(nn.Softmax(dim=-1))
            self.actor = nn.Sequential(*layers)

        layers = [nn.Linear(input_dim, critic_layer[0]),
                  nn.ReLU(inplace=True)]
        for i in range(len(critic_layer) - 1):
            layers.append(nn.Linear(critic_layer[i], critic_layer[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(critic_layer[-1], 1))
        self.critic = nn.Sequential(*layers)
        self.__network_init()

    def forward(self):
        raise NotImplementedError

    def __network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def act(self, state):
        if self.encoder is not None:
            state = self.encoder(state)
        if self.continuous:
            mu = self.actor(state).cpu()
            cov_mat = torch.diag(self.action_var)
            dist = MultivariateNormal(mu, cov_mat)
        else:
            action_prob = self.actor(state)
            dist = Categorical(action_prob)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        return action.detach(), action_log_prob.detach()

    def evaluate_AC(self, state, action):
        if self.encoder is not None:
            state = self.encoder(state)
        if self.continuous:
            mu = self.actor(state).cpu()
            action_var = self.action_var.expand_as(mu)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(mu, cov_mat)
        else:
            action_prob = self.actor(state)
            dist = Categorical(action_prob)
        action_log_prob = dist.log_prob(action.cpu())
        dist_entropy = dist.entropy()
        value = self.critic(state)
        return action_log_prob.cuda(), value, dist_entropy.cuda()

class MLPEncoder(nn.Module):
    def __init__(self, state_dim, layer_dim: list, feature_dim: int):
        super(MLPEncoder, self).__init__()
        layers = [nn.Linear(state_dim, layer_dim[0]),
                  # nn.BatchNorm1d(layer_dim[0]),
                  nn.ReLU(inplace=True)]
        for i in range(len(layer_dim) - 1):
            layers.append(nn.Linear(layer_dim[i], layer_dim[i + 1]))
            # layers.append(nn.BatchNorm1d(layer_dim[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(layer_dim[-1], feature_dim, bias=False))
        self.encoder = nn.Sequential(*layers)
        self.out_dim = feature_dim

    def forward(self, x):
        return self.encoder(x)

    def get_dim(self):
        return self.out_dim

class CNNEncoder(nn.Module):
    def __init__(self, layer_dim: list, feature_dim: list):
        super(CNNEncoder, self).__init__()
        layers = []
        for layer in layer_dim:
            layers.append(nn.Conv2d(layer[0], layer[1], layer[2], layer[3]))
            # layers.append(nn.BatchNorm2d(layer[1]))
            layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*layers)
        self.projector = nn.Linear(feature_dim[0], feature_dim[1], bias=False)
        self.out_dim = feature_dim[1]

    def forward(self, x):
        f = self.encoder(x)
        f = f.reshape(f.shape[0], -1)
        f = self.projector(f)
        return f

    def get_dim(self):
        return self.out_dim

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, q_layer, encoder=None, encoder_layer=None, feature_dim=None):
        super(QNet, self).__init__()
        self.encoder = encoder
        if encoder is None:
            input_dim = state_dim
        elif encoder == "mlp":
            self.encoder = MLPEncoder(state_dim, encoder_layer, feature_dim)
            input_dim = self.encoder.get_dim()
        elif encoder == "cnn":
            self.encoder = CNNEncoder(encoder_layer, feature_dim)
            input_dim = self.encoder.get_dim()
        else:
            raise NotImplementedError

        layers = [nn.Linear(input_dim, q_layer[0]),
                  nn.ReLU(inplace=True)]
        for i in range(len(q_layer)-1):
            layers.append(nn.Linear(q_layer[i], q_layer[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(q_layer[-1], action_dim))
        self.q_net = nn.Sequential(*layers)
        self.__network_init()

    def __network_init(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.zero_()

    def forward(self, state):
        if self.encoder is not None:
            state = self.encoder(state)
        q = self.q_net(state)
        return q

class BaseBuffer:
    def __init__(self, trans, max_len):
        self.trans = trans
        self.max_len = max_len
        self.data = {}
        for key in trans:
            self.data[key] = deque(maxlen=self.max_len)
        self.total_idx = 0

    def get_len(self):
        return self.total_idx

    def clear(self):
        """
        clear the buffer
        :return:
        """
        self.data = {}
        for key in self.trans:
            self.data[key] = []
        self.total_idx = 0

    def add(self, transition):
        """
        add a transition in buffer
        :return:
        """
        for key in transition:
            self.data[key].append(transition[key])
        self.total_idx += 1

    def get_data_buffer(self):
        data_size = len(self.data["state"])
        data = {}
        for key in self.data:
            data[key] = list(self.data[key])
        return data, data_size

    def sample(self, size):
        data_size = len(self.data["state"])
        size = min(data_size, size)
        indices = torch.randperm(data_size)[:size]
        data = {}
        for key in self.trans:
            data[key] = []
            for idx, ind in enumerate(indices):
                data[key].append(self.data[key][ind])
        return data


def unique(lists):  
    #delete duplicate attribute values
    lists = list(map(lambda x: x.lower(), lists ))
    if lists[0]=='':
        res = []
    else: 
        res = [lists[0]]
    for i in range(len(lists)):
        if i==0 or (lists[i] in lists[0:i]) or lists[i]=='':
            continue
        else:
            res.append(lists[i])
    return res

def construct_graph_from_df(df, G=None):
    # construct graph according to df
    if G is None:
        G = nx.Graph()
    for _, row in df.iterrows():
        tid = 't_' + str(row['tweet_id'])
        G.add_node(tid)

        user_ids = row['user_mentions']
        user_ids.append(row['user_id'])
        user_ids = ['u_' + str(each) for each in user_ids]
        G.add_nodes_from(user_ids)

        words = row['filtered_words']
        words = [('w_' + each).lower() for each in words]
        G.add_nodes_from(words)

        hashtags = row['hashtags']
        hashtags = [('h_' + each).lower() for each in hashtags]
        G.add_nodes_from(hashtags)

        edges = []
        #Connect the message node with each related user node, word node, etc
        edges += [(tid, each) for each in user_ids] 
        edges += [(tid, each) for each in words]
        edges += [(tid, each) for each in hashtags]
        G.add_edges_from(edges)
    return G

def construct_graph(data,feature,index):
#Build graph for a single tweet 
    G = nx.Graph()
    X = []

    tweet = data["text"].values
    X.append(feature[index].tolist())
    index = index+1
    tweet_id = data["tweet_id"].values
    G.add_node(tweet_id[0])

    user_loc = data["user_loc"].values

    f_w = data["filtered_words"].tolist()
    edges = []

    h_t = data["hashtags"].tolist()
    h_t = h_t[0]
    n = [user_loc[0]] + f_w[0] + h_t 
    n = unique(n)
    
    if len(n)!=0:
        for each in n:
            X.append(feature[index].tolist())
            index = index+1
        G.add_nodes_from(n)
        edges +=[(tweet_id[0], each) for each in n]
    G.add_edges_from(edges)
    return G,X

def normalize_adj(adj):
    # Symmetrically normalize adjacency matrix
    adj = sp.coo_matrix(adj) 
    rowsum = np.array(adj.sum(1)) 
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() 
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0. 
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) 
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() 

def aug_edge(adj): #  edge perturbation
    adj = np.array(adj)
    aug_adj1 = np.array([[i for i in j] for j in adj])
    aug_adj2 = np.array([[i for i in j] for j in adj])
    p = np.random.randint(0,len(adj)-1)
    aug_adj1[p][0] = 0
    aug_adj1[0][p] = 0
    t = np.random.randint(1,len(adj)-1)
    aug_adj1[t][p] = 1
    aug_adj1[p][t] = 1
    
    p = np.random.randint(0,len(adj)-1)
    aug_adj2[p][0] = 0
    aug_adj2[0][p] = 0
    t = np.random.randint(1,len(adj)-1)
    aug_adj2[t][p] = 1
    aug_adj2[p][t] = 1
        
    return aug_adj1,aug_adj2

def get_edge_index(adj):  #Get edge set according to adjacency matrix

    edge_index1 = []
    edge_index2 = []
    for i in range(len(adj)):
        for j in range(len(adj)):
            if adj[i][j]==1 and i<j:
                edge_index1.append(i)
                edge_index2.append(j)

    edge_index = [edge_index1] + [edge_index2]
    
    return edge_index

def get_data(message_num,start,tweet_sum,save_path):

    os.makedirs(save_path, exist_ok=True)
    df = Event2012_Dataset.load_data()
    df = df.sort_values(by='created_at').reset_index()
    ini_df = df[start:tweet_sum]

    G = construct_graph_from_df(ini_df)

    d_features = documents_to_features(df)
    print("Document features generated.")
    t_features = df_to_t_features(df)
    print("Time features generated.")
    combined_features = np.concatenate((d_features, t_features), axis=1)
    print("Concatenated document features and time features.")
    np.save(save_path + 'features_69612_0709_spacy_lg_zero_multiclasses_filtered.npy', combined_features)
    print("Initial features saved.")

    combined_features = np.load(save_path + 'features_69612_0709_spacy_lg_zero_multiclasses_filtered.npy')
    A = nx.adjacency_matrix(G).todense().tolist()
    
    X = []
    nodes = list(G.nodes)
    
    tweet=[]
    j = 0

    for i in range(len(nodes)):
        t=nodes[i][0:2]
        e=nodes[i][2:]
        if t=="t_":
            tweet.append(i)
            index=list(ini_df["tweet_id"]).index(int(e))
            X.append(list(combined_features[index]))
            j=j+1
    X = torch.tensor(X)
    adj = np.array([[0]*len(tweet)]*len(tweet))

    for i in range(len(tweet)):
        for j in range(len(A)):
            if A[tweet[i]][j]==1:
                for s in range(len(tweet)):
                    if A[j][tweet[s]]==1 and s!=i:
                        adj[i][s] = 1
    edge_index = get_edge_index(adj)

    edge_index1 = copy.deepcopy(edge_index)
    edge_index2 = copy.deepcopy(edge_index)
    true_y = torch.tensor(list(ini_df['event_id']))

    drop_percent = 0.2
    i = 0
    while 1:
        if i >= len(G.edges)*drop_percent:
            break
        m1 = random.randint(0, len(edge_index1[0])-1)
        m2 = random.randint(0, len(edge_index2[0])-1)
        if m1==m2:
            continue
        else:
            del edge_index1[0][m1]
            del edge_index1[1][m1]
            del edge_index2[0][m2]
            del edge_index2[1][m2]
    
        i = i + 1
    edge_index = torch.tensor(edge_index)
    edge_index1 = torch.tensor(edge_index1)
    edge_index2 = torch.tensor(edge_index2)

    dict_graph = {}

    dict_graph['x'] = X
    dict_graph['x1'] = X
    dict_graph['x2'] = X
    dict_graph['edge_index'] = edge_index
    dict_graph['edge_index1'] = edge_index1
    dict_graph['edge_index2'] = edge_index2
    dict_graph['y'] = true_y
    return dict_graph

def getData(M_num):  #construct an entire graph within a block
#     print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    args = args_define.args
    M =[20254,28976,30467,32302,34312,36146,37422,42700,44260,45623,46719,
        47951,51188,53160,56116,58665,59575,62251,64138,65537,66430,68840]  

    if M_num == 0:
        num = 0
        size = 500
    elif M_num == 1:
        num = M[M_num-1]
        size = 500
    elif M[M_num]-M[M_num-1]>2000:
        num = M[M_num-1]
        size = 1000
    else:
        num = M[M_num-1]
        size = M[M_num]-M[M_num-1]
    data = []
    i = M_num
    j = 0

    while 1:
        if (num+size)>=M[i]:
            tmp = get_data(i, num ,M[i], args.file_path)
            data.append(tmp)
            break
        else:
            tmp = get_data(i, num, num+size, args.file_path)
            data.append(tmp)
            j = j + 1
            print("***************Block "+str(j)+" is done.****************")
            num = num+size
    

    save_data(data, args.file_path, M_num)
    return data

def save_data(data, save_path, M_num):
    os.makedirs(save_path, exist_ok=True)
    file_path = os.path.join(save_path, f'data_{M_num}.pkl')
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved at {file_path}")


def get_Graph_Dataset(message_number):
    args = args_define.args
    print("\nBuilding graph-level social network...")
    start_time = time.time() 
    #load data for graph-level contrastive learning
    dataset = []
    label = []
    file_name = args.file_path + 'GCL-data/message_block_'+str(message_number)+'.npy'
    data = np.load(file_name,allow_pickle=True)

    for dict_data in data:
        data = Data(x=dict_data['X'],x1=dict_data['x1'],x2=dict_data['x2'],
                    edge_index=dict_data['edge_index'],edge_index1=dict_data['edge_index1'],
                    edge_index2=dict_data['edge_index2'])
        dataset.append(data)
        label.append(dict_data['label'])
    if message_number == 0 :
        dataset = loader.DataLoader(dataset,batch_size=4096)
    else:
        dataset = loader.DataLoader(dataset,batch_size=len(dataset))
    end_time = time.time()
    run_time = end_time - start_time
    print("Done! It takes "+str(int(run_time))+" seconds.\n")
    return dataset,label

def get_Node_Dataset(message_number):
    #load data for node-level contrastive learning
    print("\nBuilding node-level social network...")
    start_time = time.time() 
    #datas = getData(message_number)
    file_path = os.path.join(args.file_path, f'data_{message_number}.pkl')

    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
                datas = pickle.load(f)
        print("Data loaded successfully.")
        # 现在你可以使用 data 进行进一步操作
    else:
        print(f"No data file found at {file_path}")
        datas = getData(message_number)
    
    dataset = []
    labels = []
    
    for data in datas:
        dict_data = data

        dict_data['x'] = torch.tensor(np.array(dict_data['x']))
        dict_data['x1'] = torch.tensor(np.array(dict_data['x1']))
        dict_data['x2'] = torch.tensor(np.array(dict_data['x2']))
        dict_data['edge_index'] = torch.tensor(np.array(dict_data['edge_index']))
        dict_data['edge_index1'] = torch.tensor(np.array(dict_data['edge_index1']))
        dict_data['edge_index2'] = torch.tensor(np.array(dict_data['edge_index2']))
        data = Data(x=dict_data['x'],x1=dict_data['x1'],x2=dict_data['x2'],
                        edge_index=dict_data['edge_index'],edge_index1=dict_data['edge_index1'],
                        edge_index2=dict_data['edge_index2'])

        label = dict_data['y']
        if len(labels)==0:
            labels = label
        else:
            labels = torch.cat([labels,label])
    
        dataset.append(data)
    end_time = time.time()
    run_time = end_time - start_time
    print("Done! It takes "+str(int(run_time))+" seconds.\n")
    return dataset,np.array(labels).tolist()

# Calculate the embeddings of all the documents in the dataframe, 
# the embedding of each document is an average of the pre-trained embeddings of all the words in it
def documents_to_features(df):
    nlp = en_core_web_lg.load()
    features = df.filtered_words.apply(lambda x: nlp(' '.join(x)).vector).values
    return np.stack(features, axis=0)

# encode one times-tamp
# t_str: a string of format '2012-10-11 07:19:34'
def extract_time_feature(t_str):
    t = datetime.fromisoformat(str(t_str))
    OLE_TIME_ZERO = datetime(1899, 12, 30)
    delta = t - OLE_TIME_ZERO
    return [(float(delta.days) / 100000.), (float(delta.seconds) / 86400)]  # 86,400 seconds in day

# encode the times-tamps of all the messages in the dataframe
def df_to_t_features(df):
    t_features = np.asarray([extract_time_feature(t_str) for t_str in df['created_at']])
    return t_features


if __name__ == "__main__":
    from data_sets import Event2012_Dataset, Event2018_Dataset, MAVEN_Dataset, Arabic_Dataset

    args = args_define.args
    dataset = Event2012_Dataset.load_data()
    hcrc = HCRC(args,dataset)


    predictions, ground_truths = hcrc.detection()  # 进行预测
    results = hcrc.evaluate(predictions, ground_truths)  # 评估模型
