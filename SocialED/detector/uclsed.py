import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
import dgl
import networkx as nx
import pandas as pd
import numpy as np
from scipy import sparse
import spacy
import en_core_web_lg
import fr_core_news_lg
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, accuracy_score
import copy
import datetime
import torch.nn.functional as F
from sklearn.cluster import KMeans
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataloader import DatasetLoader

import argparse


class args_define:
    def __init__(self, **kwargs):
        # Define default values for all parameters
        defaults = {
            'file_path': '../model/model_saved/uclsed/',
            'lang': 'French', 
            'epoch': 2,
            'batch_size': 20000,
            'neighbours_num': 80,
            'GNN_h_dim': 256,
            'GNN_out_dim': 256,
            'E_h_dim': 128,
            'use_uncertainty': True,
            'use_cuda': True,
            'gpuid': 0,
            'mode': 0,
            'mse': False,
            'digamma': True,
            'log': False
        }

        # Set attributes using kwargs with defaults
        for key, default in defaults.items():
            setattr(self, key, kwargs.get(key, default))

        # Create args namespace with all parameters
        self.args = argparse.Namespace(**{k: getattr(self, k) for k in defaults.keys()})



class UCLSED:
    def __init__(self, args, dataset):
        self.save_path = None
        self.test_indices = None
        self.val_indices = None
        self.train_indices = None
        self.mask_path = None
        self.labels = None
        self.times = None
        self.g_dict = None
        self.views = None
        self.features = None

    def preprocess(self):
        preprocessor = Preprocessor(args)
        preprocessor.construct_graph(dataset)

    def fit(self):
        parser = argparse.ArgumentParser()
        print("Using CUDA:", args.use_cuda)
        if args.use_cuda:
            torch.cuda.set_device(args.gpuid)

        self.views = ['h', 'e', 'u']
        self.g_dict, self.times, self.features, self.labels = get_dgl_data(self.views)
        self.mask_path = f"{args.file_path}{args.lang}/" + "masks/"
        if not os.path.exists(self.mask_path):
            os.mkdir(self.mask_path)
        self.train_indices, self.val_indices, self.test_indices = ava_split_data(len(self.labels), self.labels,
                                                                                 len(set(self.labels)))
        torch.save(self.train_indices, self.mask_path + "train_indices.pt")
        torch.save(self.val_indices, self.mask_path + "val_indices.pt")
        torch.save(self.test_indices, self.mask_path + "test_indices.pt")

        if args.mode == 0:
            flag = ''
            if args.use_uncertainty:
                print("use_uncertainty")
                flag = "evi"
            self.save_path = f"{args.file_path}{args.lang}/" + flag + "/"
            print(self.save_path)
            os.makedirs(self.save_path, exist_ok=True)
        else:
            self.save_path = '../model/model_saved/uclsed/Eng_CrisisLexT26/'

        if args.use_uncertainty:
            if args.digamma:
                criterion = edl_digamma_loss
            elif args.log:
                criterion = edl_log_loss
            elif args.mse:
                criterion = edl_mse_loss
            else:
                parser.error("--uncertainty requires --mse, --log or --digamma.")
        else:
            criterion = nn.CrossEntropyLoss()

        self.model = UCLSED_model(self.features.shape[1], args.GNN_h_dim, args.GNN_out_dim, args.E_h_dim,
                                  len(set(self.labels)), self.views)
        self.model = train_model(self.model, self.g_dict, self.views, self.features, self.times, self.labels,
                                 args.epoch, criterion, self.mask_path, self.save_path, args)

    def detection(self):
        self.model.eval()
        self.val_indices = torch.load(self.mask_path + "val_indices.pt")
        classes = len(set(self.labels))
        self.labels = make_onehot(self.labels, classes)
        device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
        if args.use_cuda:
            self.model = self.model.cuda()
            self.features = self.features.cuda()
            self.times = self.times.cuda()
            self.labels = self.labels.cuda()
            self.train_indices = self.train_indices.cuda()
            self.test_indices = self.test_indices.cuda()
            self.val_indices = self.val_indices.cuda()
            for v in self.views:
                self.g_dict[v] = self.g_dict[v].to(device)
                self.g_dict[v].ndata['features'] = self.features
                self.g_dict[v].ndata['t'] = self.times

        out, emb, nids = extract_results(self.g_dict, self.views, self.labels, self.model, args)
        ori_labels = self.labels
        # extract_labels = ori_labels[nids]
        extract_labels = ori_labels[nids].cpu()
        
        comb_out = None
        if args.use_uncertainty:
            alpha = []
            for out_v in out.values():
                evi_v = relu_evidence(out_v)
                alpha_v = evi_v + 1
                alpha.append(alpha_v)
            comb_out, comb_u = DS_Combin(alpha=alpha, classes=classes)

        else:
            for i, out_v in enumerate(out.values()):
                if i == 0:
                    comb_out = out_v
                else:
                    comb_out += out_v

        _, val_pred = torch.max(comb_out[self.val_indices.cpu().numpy()], 1)
        #val_labels = torch.IntTensor(extract_labels[self.val_indices.cpu().numpy()])
        val_labels = torch.argmax(extract_labels[self.val_indices.cpu().numpy()], 1)
        predictions = val_pred.cpu().numpy()
        ground_truth = val_labels.cpu().numpy()

        return ground_truth, predictions

    def evaluate(self, ground_truth, predictions):
        val_f1 = f1_score(ground_truth, predictions, average='macro')
        val_acc = accuracy_score(ground_truth, predictions)

        print(f"Validation F1 Score: {val_f1}")
        print(f"Validation Accuracy: {val_acc}")




class Preprocessor:
    def __init__(self, dataset):
        pass

    def str2list(self, str_ele):
        if str_ele == "[]":
            value = []
        else:
            value = [e.replace('\'', '').lstrip().replace(":", '') for e in str(str_ele)[1:-1].split(',') if
                     len(e.replace('\'', '').lstrip().replace(":", '')) > 0]
        return value

    def load_data(self, dataset):
        ori_df = dataset

        ori_df.drop_duplicates(["tweet_id"], keep='first', inplace=True)
        event_id_num_dict = {}
        select_index_list = []

        for id in set(ori_df["event_id"]):
            num = len(ori_df.loc[ori_df["event_id"] == id])
            if int(num / 3) >= 25:
                event_id_num_dict[id] = int(num / 3 + 50)
                select_index_list += list(ori_df.loc[ori_df["event_id"] == id].index)[0:int(num / 3 + 50)]
        select_df = ori_df.loc[select_index_list]
        select_df = select_df.reset_index(drop=True)
        id_num = sorted(event_id_num_dict.items(), key=lambda x: x[1], reverse=True)

        for (i, j) in id_num[0:100]:
            print(j, end=",")
        sorted_id_dict = dict(zip(np.array(id_num)[:, 0], range(0, len(set(ori_df["event_id"])))))
        sorted_df = select_df
        sorted_df["event_id"] = sorted_df["event_id"].apply(lambda x: sorted_id_dict[x])

        print(sorted_df.shape)
        data_value = sorted_df[
            ["tweet_id", "user_mentions", "text", "hashtags", "entities", "urls", "filtered_words", "created_at",
             "event_id"]].values
        event_df = pd.DataFrame(data=data_value,
                                columns=["tweet_id", "mention_user", "text", "hashtags", "entities", "urls",
                                         "filtered_words", "timestamp", "event_id"])
        event_df['hashtags'] = event_df['hashtags'].apply(lambda x: ["h_" + i for i in x])
        event_df['entities'] = event_df['entities'].apply(lambda x: ["e_" + str(i) for i in x])
        event_df['mention_user'] = event_df['mention_user'].apply(lambda x: ["u_" + str(i) for i in x])
        event_df = event_df.loc[event_df['event_id'] < 100]
        event_df = event_df.reset_index(drop=True)

        print(event_df.shape)
        return event_df

    def get_nlp(self, lang):
        if lang == "English" or lang == "Arabic":
            # nlp = en_core_web_lg.load()
            nlp =spacy.load('en_core_web_lg')
        elif lang == "French":
            # nlp = fr_core_news_lg.load()
            nlp=spacy.load('fr_core_news_lg')
        return nlp

    def construct_graph_base_eles(self, view_dict, df, path, lang):
        os.makedirs(path, exist_ok=True)
        nlp = self.get_nlp(lang)
        df = df.drop_duplicates(subset=['tweet_id'])
        df.reset_index()
        df.drop_duplicates(["tweet_id"], keep='first', inplace=True)
        print("generate text features---------")
        features = np.stack(df['filtered_words'].apply(lambda x: nlp(' '.join(x)).vector).values, axis=0)
        print(features.shape)
        np.save(path + "features.npy", features)
        print("text features are saved in {}features.npy".format(path))
        np.save(path + "time.npy", df['timestamp'].values)
        print("time features are saved in {}time.npy".format(path))
        df["event_id"] = df["event_id"].apply(lambda x: int(x))
        np.save(path + "label.npy", df['event_id'].values)
        print("labels are saved in {}label.npy".format(path))

        true_matrix = np.eye(df.shape[0])
        for i in range(df.shape[0]):
            label_i = df["event_id"].values[i]
            indices = df[df["event_id"] == label_i].index
            true_matrix[i, indices] = 1
        # print(true_matrix)

        print("construct graph---------------")
        G = nx.Graph()
        for _, row in df.iterrows():
            tid = str(row['tweet_id'])
            G.add_node(tid)
            G.nodes[tid]['tweet_id'] = True  # right-hand side value is irrelevant for the lookup
            edges = []
            for view in view_dict.values():
                for ele in view:
                    if len(row[ele]) > 0:
                        ele_values = row[ele]
                        G.add_nodes_from(ele_values)
                        for each in ele_values:
                            G.nodes[each][ele] = True
                        edges += [(tid, each) for each in row[ele]]

            G.add_edges_from(edges)

        all_nodes = list(G.nodes)
        matrix = nx.to_scipy_sparse_array(G)
        tweet_nodes = list(nx.get_node_attributes(G, "tweet_id").keys())
        # print(tweet_nodes)
        print(len(tweet_nodes))
        tweet_index = [all_nodes.index(t_node) for t_node in tweet_nodes]

        for v, view in zip(view_dict.keys(), view_dict.values()):
            s_tweet_tweet_matrix = sparse.csr_matrix(np.identity(len(tweet_nodes)))
            for ele in view:
                ele_nodes = list(nx.get_node_attributes(G, ele).keys())
                ele_index = [all_nodes.index(e_node) for e_node in ele_nodes]
                tweet_ele_matrix = matrix[tweet_index, :][:, ele_index]
                s_ele_tweet_tweet_matrix = sparse.csr_matrix(tweet_ele_matrix @ tweet_ele_matrix.transpose())
                s_tweet_tweet_matrix += s_ele_tweet_tweet_matrix
            s_tweet_tweet_matrix = s_tweet_tweet_matrix.astype('bool')
            sparse.save_npz(os.path.join(path, f"s_tweet_tweet_matrix_{v}.npz"), s_tweet_tweet_matrix)
            print(f"Sparse binary {v} commuting matrix is saved in {path}s_tweet_tweet_matrix_{v}.npz")

    def construct_graph(self, dataset):
        event_df = self.load_data(dataset)
        view_dict = {"h": ["hashtags", "urls"], "u": ["mention_user"], "e": ["entities"]}
        path = args.file_path + args.lang + '/'
        self.construct_graph_base_eles(view_dict, event_df, path, args.lang)


def extract_results(g_dict, views, labels, model, args, train_indices=None):
    with torch.no_grad():
        model.eval()
        out_list = []
        emb_list = []
        nids_list = []
        all_indices = torch.LongTensor(range(0, labels.shape[0]))
        if args.use_cuda:
            all_indices = all_indices.cuda()
        print(all_indices)
        sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
        dataloader = dgl.dataloading.NodeDataLoader(
            g_dict[views[0]], all_indices, sampler,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )

        for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
            device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
            extract_indices = blocks[-1].dstdata[dgl.NID].to(device)
            blocks_dict = {}
            blocks_dict[views[0]] = blocks
            for v in views[1:]:
                blocks_v = list(dgl.dataloading.NodeDataLoader(
                    g_dict[v], extract_indices, sampler,
                    batch_size=args.batch_size,
                    shuffle=False,
                    drop_last=False,
                ))[0][2]
                blocks_dict[v] = blocks_v
            for v in views:
                blocks_dict[v] = [b.to(device) for b in blocks_dict[v]]

            # 将 blocks_dict 中的所有 blocks 移动到相同的设备
            # blocks_dict = {v: [b.to(device) for b in blocks] for v, blocks in blocks_dict.items()}
            # extract_indices = extract_indices.to(device)

            out, emb = model(blocks_dict)
            out_list.append(out)
            emb_list.append(emb)
            nids_list.append(extract_indices)

    # assert batch_id==0
    all_out = {}
    all_emb = {}
    for v in views:
        all_out[v] = []
        all_emb[v] = []
        for out, emb in zip(out_list, emb_list):
            all_out[v].append(out[v])
            all_emb[v].append(emb[v])
        if args.use_cuda:
            all_out[v] = torch.cat(all_out[v]).cpu()
            all_emb[v] = torch.cat(all_emb[v]).cpu()
        else:
            all_out[v] = torch.cat(all_out[v])
            all_emb[v] = torch.cat(all_emb[v])

    extract_nids = torch.cat(nids_list)
    if args.use_cuda:
        extract_nids = extract_nids.cpu()

    return all_out, all_emb, extract_nids


def train_model(model, g_dict, views, features, times, labels, epoch, criterion, mask_path, save_path, args):
    train_indices = torch.load(mask_path + "train_indices.pt")
    val_indices = torch.load(mask_path + "val_indices.pt")
    test_indices = torch.load(mask_path + "test_indices.pt")
    classes = len(set(labels))
    ori_labels = labels

    labels = make_onehot(labels, classes)
    device = torch.device("cuda:{}".format(args.gpuid) if args.use_cuda else "cpu")
    if args.use_cuda:
        model = model.cuda()
        features = features.cuda()
        times = times.cuda()
        labels = labels.cuda()
        train_indices = train_indices.cuda()
        val_indices = val_indices.cuda()
        test_indices = test_indices.cuda()

    for v in views:
        if args.use_cuda:
            g_dict[v] = g_dict[v].to(device)
        g_dict[v].ndata['features'] = features
        g_dict[v].ndata['t'] = times

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=0.005)
    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    if args.mode == 0:
        message = "----------begin training---------\n"
        with open(save_path + "log.txt", 'w') as f:
            f.write(message)

        best_vali = 0
        test_acc_in_best_e = 0
        best_epoch = 0
        test_acc_list = []
        label_u = torch.FloatTensor(np.ones(classes))

        for e in range(epoch):
            print(f"Epoch {e + 1}/{epoch}")

            _, GNN_out_fea, extract_nids = extract_results(g_dict, views, labels, model, args)

            for v in GNN_out_fea:
                GNN_out_fea[v] = GNN_out_fea[v].to(device)
                # print(f'GNN_out_fea[{v}].device: {GNN_out_fea[v].device}')  # 确认设备

            extract_labels = ori_labels[extract_nids]
            label_center = {}
            for v in views:
                label_center[v] = []
            for l in range(classes):
                l_indices = torch.LongTensor(np.where(extract_labels == l)[0].reshape(-1)).to(device)
                # print(l_indices.device)
                for v in views:
                    # print(f'GNN_out_fea[{v}].device:{GNN_out_fea[v].device}')
                    # print(f'l_indices.device:{l_indices.device}')

                    l_feas = GNN_out_fea[v][l_indices]
                    l_cen = torch.mean(l_feas, dim=0)
                    label_center[v].append(l_cen)

            for v in views:
                label_center[v] = torch.stack(label_center[v], dim=0)
                label_center[v] = F.normalize(label_center[v], 2, 1)

                if args.use_cuda:
                    label_center[v] = label_center[v].cuda()
                    label_u = label_u.cuda()

            losses = []
            total_loss = 0
            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
            dataloader = dgl.dataloading.NodeDataLoader(
                g_dict[views[0]], train_indices, sampler,
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                device=device
            )

            print(f"Dataloader initialized with {len(dataloader)} batches")
            for batch_id, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
                print(f"Processing batch {batch_id + 1}/{len(dataloader)}")
                batch_indices = blocks[-1].dstdata[dgl.NID]
                if args.use_cuda:
                    batch_indices = batch_indices.cuda()
                blocks_dict = {}
                blocks_dict[views[0]] = blocks
                for v in views[1:]:
                    blocks_v = list(dgl.dataloading.NodeDataLoader(
                        g_dict[v], batch_indices, sampler,
                        batch_size=args.batch_size,
                        shuffle=False,
                        drop_last=False,
                    ))[0][2]
                    blocks_dict[v] = blocks_v

                for v in views:
                    blocks_dict[v] = [b.to(device) for b in blocks_dict[v]]

                batch_labels = labels[batch_indices]
                batch_ori_labels = torch.LongTensor(ori_labels).to(device)[batch_indices]
                model.train()
                out, emb = model(blocks_dict)

                view_contra_loss = 0
                e_loss = 0
                if args.use_uncertainty:
                    alpha = []
                    true_labels = torch.LongTensor(ori_labels).to(device)[batch_indices]
                    for i, v in enumerate(views):
                        emb[v] = F.normalize(emb[v], 2, 1)
                        batch_center = label_center[v][batch_ori_labels]

                        view_contra_loss += torch.mean(-torch.log(
                            (torch.exp(torch.sum(torch.mul(emb[v], batch_center), dim=1)) - 0.1 * label_u[
                                batch_ori_labels]) / (
                                torch.sum(torch.exp(torch.mm(emb[v], label_center[v].T)),
                                          dim=1))))  # *label_u[batch_ori_labels])

                        alpha_v = relu_evidence(out[v]) + 1
                        alpha.append(alpha_v)

                    comb_alpha, comb_u = DS_Combin(alpha=alpha, classes=classes)

                    e_loss = EUC_loss(comb_alpha, comb_u, true_labels, e)
                    loss = e_loss + criterion(comb_alpha, batch_labels, true_labels, e, classes, 100,
                                              device) + 2 * view_contra_loss

                else:
                    batch_labels = torch.argmax(batch_labels, 1)
                    for i, v in enumerate(views):
                        if i == 0:
                            comb_out = out[v]
                        else:
                            comb_out += out[v]
                        emb[v] = F.normalize(emb[v], 2, 1)
                        batch_center = label_center[v][batch_ori_labels]
                        view_contra_loss += torch.mean(-torch.log(
                            (torch.exp(torch.sum(torch.mul(emb[v], batch_center), dim=1))) / (
                                torch.sum(torch.exp(torch.mm(emb[v], label_center[v].T)), dim=1))))
                    loss = criterion(comb_out, batch_labels)  # + view_contra_loss

                com_loss = 0
                for i in range(len(emb) - 1):
                    for j in range(i + 1, len(emb)):
                        com_loss += common_loss(emb[views[i]], emb[views[j]])
                loss += 1 * com_loss
                print("com_loss:", 1 * com_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
                total_loss += loss.item()
                print(loss)
                print("Batch loss:", loss.item())

            total_loss /= (batch_id + 1)
            message = 'Epoch: {}/{}. Average loss: {:.4f}'.format(e + 1, args.epoch, total_loss)
            print(message)
            with open(save_path + '/log.txt', 'a') as f:
                f.write(message)
                f.write("\n")
            out, emb, nids = extract_results(g_dict, views, labels, model, args)
            # nids = torch.cat(nids).cpu().numpy().astype(int)  # 确保 nids 是整数数组
            extract_labels = ori_labels[nids]
            if args.use_uncertainty:
                alpha = []
                for out_v in out.values():
                    evi_v = relu_evidence(out_v)
                    alpha_v = evi_v + 1
                    alpha.append(alpha_v)
                comb_out, comb_u = DS_Combin(alpha=alpha, classes=classes)
                train_labels = extract_labels[train_indices.cpu().numpy()]

                comb_u = comb_u.cuda()
                train_u = comb_u[train_indices].cpu().numpy()
                train_i_u = []
                for i in range(classes):
                    i_indices = np.where(train_labels == i)
                    i_u = np.mean(train_u[i_indices])
                    train_i_u.append(i_u)
                label_u = torch.FloatTensor(train_i_u).cuda()
                # print("label_u:",label_u)

            else:
                for i, out_v in enumerate(out.values()):
                    if i == 0:
                        comb_out = out_v
                    else:
                        comb_out += out_v

            _, val_pred = torch.max(comb_out[val_indices.cpu().numpy()], 1)
            val_labels = torch.IntTensor(extract_labels[val_indices.cpu().numpy()])
            val_f1 = f1_score(val_labels.cpu().numpy(), val_pred.cpu().numpy(), average='macro')
            val_match = torch.reshape(torch.eq(val_pred, val_labels).float(), (-1, 1))
            val_acc = torch.mean(val_match)

            _, test_pred = torch.max(comb_out[test_indices.cpu().numpy()], 1)
            test_labels = torch.IntTensor(extract_labels[test_indices.cpu().numpy()])
            test_f1 = f1_score(test_labels.cpu().numpy(), test_pred.cpu().numpy(), average='macro')
            test_match = torch.reshape(torch.eq(test_pred, test_labels).float(), (-1, 1))
            test_acc = torch.mean(test_match)
            # t = classification_report(test_labels.cpu().numpy(), test_pred.cpu().numpy(), target_names=[i for i in range(classes)])
            message = "val_acc: %.4f val_f1:%.4f  test_acc: %.4f test_f1:%.4f" % (val_acc, val_f1, test_acc, test_f1)
            print(message)
            with open(save_path + '/log.txt', 'a') as f:
                f.write(message)

            test_acc_list.append(test_acc)

            if val_acc > best_vali:
                best_vali = val_acc
                best_epoch = e + 1
                test_acc_in_best_e = test_acc
                p = save_path + 'best.pt'
                torch.save(model.state_dict(), p)

        np.save(save_path + "testacc.npy", np.array(test_acc_list))
        message = "best epoch:%d  test_acc:%.4f" % (best_epoch, test_acc_in_best_e)
        print(message)
        with open(save_path + '/log.txt', 'a') as f:
            f.write(message)

    else:
        model.load_state_dict(torch.load(save_path + '/best.pt'))
        model.eval()
        out, emb, nids = extract_results(g_dict, views, labels, model, args)
        extract_labels = ori_labels[nids]
        if args.use_uncertainty:
            alpha = []
            for v in ['h', 'u', 'e']:
                evi_v = relu_evidence(out[v])
                alpha_v = evi_v + 1
                alpha.append(alpha_v)
            comb_out, comb_u = DS_Combin(alpha=alpha, classes=classes)
        else:
            for i, v in enumerate(views):
                if i == 0:
                    comb_out = out[v]
                else:
                    comb_out += out[v]

        _, test_pred = torch.max(comb_out[test_indices.cpu().numpy()], 1)
        test_labels = torch.IntTensor(extract_labels[test_indices.cpu().numpy()])
        if args.use_uncertainty:
            test_u = comb_u[test_indices].cpu().numpy()
            test_match = torch.reshape(torch.eq(test_pred.cpu().numpy(), test_labels.cpu().numpy()).float(), (-1, 1))
            test_i_u = []
            for i in range(classes):
                i_indices = np.where(test_labels.cpu().numpy() == i)
                i_u = np.mean(test_u[i_indices])
                test_i_u.append(i_u)

        test_f1 = f1_score(test_labels.cpu().numpy(), test_pred.cpu().numpy(), average='macro')
        test_match = torch.reshape(torch.eq(test_pred.cpu().numpy(), test_labels.cpu().numpy()).float(), (-1, 1))
        test_acc = torch.mean(test_match)
        t = classification_report(test_labels.cpu().numpy(), test_pred.cpu().numpy())
        message = "test_acc: %.4f test_f1:%.4f" % (test_acc, test_f1)
        print(message)

    return model


class Tem_Agg_Layer(nn.Module):
    def __init__(self, in_dim, out_dim, use_residual):
        super(Tem_Agg_Layer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.temporal_fc = nn.Linear(out_dim, 1, bias=False)
        self.reset_parameters()
        self.use_residual = use_residual

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)

    def edge_attention(self, edges):
        deltas = edges.src['t'] - edges.dst['t']
        deltas = deltas.cpu().detach().numpy()
        weights = -abs(deltas)
        return {'e': torch.tensor(weights).unsqueeze(1).to(edges.src['t'].device)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(torch.exp(self.temporal_fc(nodes.mailbox['z']) * nodes.mailbox['e'] / 500), dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, blocks, layer_id):
        device = blocks[layer_id].device  # 获取当前block的设备
        h = blocks[layer_id].srcdata['features'].to(device)
        z = self.fc(h)
        blocks[layer_id].srcdata['z'] = z
        z_dst = z[:blocks[layer_id].number_of_dst_nodes()]

        blocks[layer_id].dstdata['z'] = z_dst
        blocks[layer_id].apply_edges(self.edge_attention)

        blocks[layer_id].update_all(self.message_func, self.reduce_func)

        if self.use_residual:
            return z_dst + blocks[layer_id].dstdata['h']
        return blocks[layer_id].dstdata['h']


class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, use_residual=False):
        super(GNN, self).__init__()
        self.layer1 = Tem_Agg_Layer(in_dim, hidden_dim, use_residual)
        self.layer2 = Tem_Agg_Layer(hidden_dim, out_dim, use_residual)

    def forward(self, blocks):
        device = blocks[0].device  # 获取第一个block的设备
        self.layer1 = self.layer1.to(device)
        self.layer2 = self.layer2.to(device)

        h = self.layer1(blocks, 0)
        h = F.elu(h)
        blocks[1].srcdata['features'] = h.to(device)
        h = self.layer2(blocks, 1)
        return h

    def edge_attention(self, edges):
        device = edges.data['features'].device
        return self.calculate_attention(edges).to(device)

    def calculate_attention(self, edges):
        # edge attention的计算逻辑
        pass


class EDNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, use_dropout=True):
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


class UCLSED_model(nn.Module):
    def __init__(self, GNN_in_dim, GNN_h_dim, GNN_out_dim, E_h_dim, E_out_dim, views):
        super(UCLSED_model, self).__init__()
        self.views = views
        self.GNN = GNN(GNN_in_dim, GNN_h_dim, GNN_out_dim)
        self.EDNNs = nn.ModuleList([EDNN(GNN_out_dim, E_h_dim, E_out_dim) for v in self.views])

    def forward(self, blocks_dict, is_EDNN_input=False, i=None, emb_v=None):
        out = dict()
        if not is_EDNN_input:
            emb = dict()
            for i, v in enumerate(self.views):
                emb[v] = self.GNN(blocks_dict[v])
                out[v] = self.EDNNs[i](emb[v])
            return out, emb
        else:
            out = self.EDNNs[i](emb_v)
            return out


# loss
def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2) ** 2)
    return cost


def EUC_loss(alpha, u, true_labels, e):
    _, pred_label = torch.max(alpha, 1)
    true_indices = torch.where(pred_label == true_labels)
    false_indices = torch.where(pred_label != true_labels)
    S = torch.sum(alpha, dim=1, keepdim=True)
    p, _ = torch.max(alpha / S, 1)
    a = -0.01 * torch.exp(-(e + 1) / 10 * torch.log(torch.FloatTensor([0.01]))).cuda()
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor((e + 1) / 10, dtype=torch.float32),
    )
    EUC_loss = -annealing_coef * torch.sum((p[true_indices] * (torch.log(1.000000001 - u[true_indices]).squeeze(
        -1))))  # -(1-annealing_coef)*torch.sum(((1-p[false_indices])*(torch.log(u[false_indices]).squeeze(-1))))

    return EUC_loss


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


def kl_pred_divergence(alpha, y, num_classes, device):
    # max_alpha, _ = torch.max(alpha, 1)
    # ones = alpha*(1-y) + (max_alpha+1) * y
    ones = y + 0.01 * torch.ones([1, num_classes], dtype=torch.float32, device=device)
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


def edl_loss(func, y, true_labels, alpha, epoch_num, num_classes, annealing_step, device):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor((epoch_num + 1) / 10, dtype=torch.float32),
    )

    _, pred_label = torch.max(alpha, 1)
    true_indices = torch.where(pred_label == true_labels)
    false_indices = torch.where(pred_label != true_labels)
    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    print("kl_div:", 1 * torch.mean(kl_div))
    print("A:", 20 * torch.mean(A))

    return 20 * A + 1 * kl_div


def edl_mse_loss(alpha, target, true_labels, epoch_num, num_classes, annealing_step, device):
    # evidence = relu_evidence(output)
    # alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, true_labels, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


def edl_log_loss(alpha, target, true_labels, epoch_num, num_classes, annealing_step, device):
    # evidence = relu_evidence(output)
    # alpha = evidence + 1
    loss = torch.mean(edl_loss(
        torch.log, target, alpha, true_labels, epoch_num, num_classes, annealing_step, device
    )
    )
    return loss


def edl_digamma_loss(alpha, target, true_labels, epoch_num, num_classes, annealing_step, device):
    # evidence = relu_evidence(output)
    # alpha = evidence + 1

    loss = torch.mean(edl_loss(
        torch.digamma, target, true_labels, alpha, epoch_num, num_classes, annealing_step, device

    ))
    return loss


# utils
def make_onehot(input, classes):
    input = torch.LongTensor(input).unsqueeze(1)
    result = torch.zeros(len(input), classes).long()
    result.scatter_(dim=1, index=input.long(), src=torch.ones(len(input), classes).long())
    return result


def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def DS_Combin(alpha, classes):
    """
    :param alpha: All Dirichlet distribution parameters.
    :return: Combined Dirichlet distribution parameters.
    """

    def DS_Combin_two(alpha1, alpha2, classes):
        """
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v] - 1
            b[v] = E[v] / (S[v].expand(E[v].shape))
            u[v] = classes / S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

        # calculate new S
        S_a = classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a, u_a

    if len(alpha) == 1:
        S = torch.sum(alpha[0], dim=1, keepdim=True)
        u = classes / S
        return alpha[0], u
    for v in range(len(alpha) - 1):
        if v == 0:
            alpha_a, u_a = DS_Combin_two(alpha[0], alpha[1], classes)
        else:
            alpha_a, u_a = DS_Combin_two(alpha_a, alpha[v + 1], classes)
    return alpha_a, u_a


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
    with open(save_path + "/graph_statistics.txt", "w") as f:
        f.write(message)
    return num_isolated_nodes


def get_dgl_data(views):
    g_dict = {}
    path = args.file_path + args.lang + '/'
    features = torch.FloatTensor(np.load(path + "features.npy"))
    times = np.load(path + "time.npy")
    times = torch.FloatTensor(((times - times.min()).astype('timedelta64[D]') / np.timedelta64(1, 'D')))
    labels = np.load(path + "label.npy")
    for v in views:
        if v == "h":
            matrix = sparse.load_npz(path + "s_tweet_tweet_matrix_{}.npz".format(v))
            # matrix = np.load(path + "matrix_{}.npy".format(v+noise))
        else:
            matrix = sparse.load_npz(path + "s_tweet_tweet_matrix_{}.npz".format(v))
        g = dgl.DGLGraph(matrix, readonly=True)
        save_path_v = path + v
        if not os.path.exists(save_path_v):
            os.mkdir(save_path_v)
        num_isolated_nodes = graph_statistics(g, save_path_v)
        g.set_n_initializer(dgl.init.zero_initializer)
        # g.readonly(readonly_state=True)
        # g.ndata['features'] = features
        # # g.ndata['labels'] = labels
        # g.ndata['times'] = times
        g_dict[v] = g
    return g_dict, times, features, labels


def split_data(length, train_p, val_p, test_p):
    indices = torch.randperm(length)
    val_samples = int(length * val_p)
    val_indices = indices[:val_samples]
    test_samples = val_samples + int(length * test_p)
    test_indeces = indices[val_samples:test_samples]
    train_indices = indices[test_samples:]
    return train_indices, val_indices, test_indeces


def ava_split_data(length, labels, classes):
    indices = torch.randperm(length)
    labels = torch.LongTensor(labels[indices])

    train_indices = []
    test_indices = []
    val_indices = []

    for l in range(classes):
        l_indices = torch.LongTensor(np.where(labels.numpy() == l)[0].reshape(-1))
        val_indices.append(l_indices[:20].reshape(-1, 1))
        test_indices.append(l_indices[20:50].reshape(-1, 1))
        train_indices.append(l_indices[50:].reshape(-1, 1))

    val_indices = indices[torch.cat(val_indices, dim=0).reshape(-1)]
    test_indices = indices[torch.cat(test_indices, dim=0).reshape(-1)]
    train_indices = indices[torch.cat(train_indices, dim=0).reshape(-1)]
    print(train_indices.shape, val_indices.shape, test_indices.shape)
    print(train_indices)
    return train_indices, val_indices, test_indices


if __name__ == "__main__":
    from dataset.dataloader import Event2012
    dataset = Event2012()
    args = args_define().args

    uclsed = UCLSED(args, dataset)
    uclsed.preprocess()
    uclsed.fit()
    predictions, ground_truths = uclsed.detection()  
    uclsed.evaluate(predictions, ground_truths) 
