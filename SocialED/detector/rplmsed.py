#!/usr/bin/env python
# encoding: utf-8
import numpy as np
import numpy
numpy._import_array()
import logging
import argparse
import gc
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.metrics import RunningAverage, Average
from sklearn.metrics import accuracy_score
from torch.optim.lr_scheduler import StepLR
from transformers import AutoTokenizer, AutoModel, AutoConfig
from collections import namedtuple, OrderedDict, Counter
from typing import Any, List
import math
import os
import random
import torch
from torch import nn
from sklearn import metrics, manifold
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from hdbscan import HDBSCAN
from matplotlib import pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import itertools
import scipy as sp
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataloader import DatasetLoader



logging.basicConfig(level=logging.WARN,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt='%Y-%m-%d  %H:%M:%S %a')


class args_define:
    def __init__(self, **kwargs):
        # Hyper parameters
        self.dataset = kwargs.get('dataset', '../model/model_saved/rplmsed/cache/twitter12.npy')
        self.plm_path = kwargs.get('plm_path', '../model/model_needed/base_plm_model/roberta-large')
        self.file_path = kwargs.get('file_path', '../model/model_saved/rplmsed/')
        self.plm_tuning = kwargs.get('plm_tuning', False)
        self.use_ctx_att = kwargs.get('use_ctx_att', False)
        self.offline = kwargs.get('offline', True)
        self.ctx_att_head_num = kwargs.get('ctx_att_head_num', 2)
        self.pmt_feats = kwargs.get('pmt_feats', (0, 1, 2, 4))
        self.batch_size = kwargs.get('batch_size', 128)
        # self.batch_size = kwargs.get('batch_size', 32)
        self.lmda1 = kwargs.get('lmda1', 0.010)
        self.lmda2 = kwargs.get('lmda2', 0.005)
        self.tao = kwargs.get('tao', 0.90)
        self.optimizer = kwargs.get('optimizer', 'Adam')
        self.lr = kwargs.get('lr', 2e-5)
        self.weight_decay = kwargs.get('weight_decay', 1e-5)
        self.momentum = kwargs.get('momentum', 0.9)
        self.step_lr_gamma = kwargs.get('step_lr_gamma', 0.98)
        self.max_epochs = kwargs.get('max_epochs', 1)
        self.ckpt_path = kwargs.get('ckpt_path', '../model/model_saved/rplmsed/ckpt/')
        self.eva_data = kwargs.get('eva_data', "../model/model_saved/rplmsed/Eva_data/")
        self.early_stop_patience = kwargs.get('early_stop_patience', 2)
        self.early_stop_monitor = kwargs.get('early_stop_monitor', 'loss')
        self.SAMPLE_NUM_TWEET = kwargs.get('SAMPLE_NUM_TWEET', 60)
        self.WINDOW_SIZE = kwargs.get('WINDOW_SIZE', 3)
        self.device = kwargs.get('device', "cuda:0" if torch.cuda.is_available() else "cpu")

        # Store all arguments in a single attribute
        self.args = argparse.Namespace(**{
            'dataset': self.dataset,
            'plm_path': self.plm_path,
            'file_path': self.file_path,
            'plm_tuning': self.plm_tuning,
            'use_ctx_att': self.use_ctx_att,
            'offline': self.offline,
            'ctx_att_head_num': self.ctx_att_head_num,
            'pmt_feats': self.pmt_feats,
            'batch_size': self.batch_size,
            'lmda1': self.lmda1,
            'lmda2': self.lmda2,
            'tao': self.tao,
            'optimizer': self.optimizer,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            'momentum': self.momentum,
            'step_lr_gamma': self.step_lr_gamma,
            'max_epochs': self.max_epochs,
            'ckpt_path': self.ckpt_path,
            'eva_data': self.eva_data,
            'early_stop_patience': self.early_stop_patience,
            'early_stop_monitor': self.early_stop_monitor,
            'SAMPLE_NUM_TWEET': self.SAMPLE_NUM_TWEET,
            'WINDOW_SIZE': self.WINDOW_SIZE,
            'device': self.device
        })



class Preprocessor:
    args = args_define().args

    def __init__(self):
        pass

    def preprocess(self, dataset):
        os.makedirs('../model/model_saved/rplmsed/cache', exist_ok=True)
        print(f"load data  ... ")
        df = dataset
        np_data = df.to_numpy()
        print("\tDone")

        blk_data = self.pre_process(np_data)
        print(f"save data to '{self.dataset}' ... ", end='')
        np.save(f'{self.dataset}', blk_data)
        print("\tDone")

    def to_sparse_matrix(self, feat_to_tw, tw_num, tao=0):
        tw_adj = sp.sparse.coo_matrix((tw_num, tw_num), dtype=np.int8)
        tw_adj = tw_adj.todok()  # convert to dok
        for f in feat_to_tw.keys():
            for i in feat_to_tw[f]:
                for j in feat_to_tw[f]:
                    tw_adj[i, j] += 1

        tw_adj = tw_adj > tao
        tw_adj = tw_adj.tocsr().astype(np.int8)
        return tw_adj

    def build_entity_adj(self, data):
        tw_num = len(data)
        feat_to_tw = {}
        for i, it in enumerate(data):
            feats = it.entities
            feats = [e for e, t in feats]

            for f in feats:
                f = f.lower()
                if f not in feat_to_tw:
                    feat_to_tw[f] = set()
                feat_to_tw[f].add(i)

        return self.to_sparse_matrix(feat_to_tw, tw_num)

    def build_hashtag_adj(self, data):
        tw_num = len(data)
        feat_to_tw = {}
        for i, it in enumerate(data):
            feats = it.hashtags

            for f in feats:
                f = f.lower()
                if f not in feat_to_tw:
                    feat_to_tw[f] = set()
                feat_to_tw[f].add(i)

        return self.to_sparse_matrix(feat_to_tw, tw_num)

    def build_words_adj(self, data):
        tw_num = len(data)
        feat_to_tw = {}
        for i, it in enumerate(data):
            feats = it.words

            for f in feats:
                f = f.lower()
                if f not in feat_to_tw:
                    feat_to_tw[f] = set()
                feat_to_tw[f].add(i)

        return self.to_sparse_matrix(feat_to_tw, tw_num)

    def build_user_adj(self, data):
        tw_num = len(data)
        feat_to_tw = {}
        for i, it in enumerate(data):
            feats = it.user_mentions
            feats.append(it.user_id)

            for f in feats:
                if f not in feat_to_tw:
                    feat_to_tw[f] = set()
                feat_to_tw[f].add(i)

        return self.to_sparse_matrix(feat_to_tw, tw_num)

    def build_creat_at_adj(self, data):
        tw_num = len(data)
        tw_feat_idx = []
        feat_to_idx = {}
        for i, it in enumerate(data):
            feats = it.created_at
            feats = [e for e, t in feats]

            for f in feats:
                if f not in feat_to_idx:
                    feat_to_idx[f] = len(feat_to_idx)
                f_idx = feat_to_idx[f]

                tw_feat_idx.append([i, f_idx])

        tw_feat_val = np.ones((len(tw_feat_idx),), dtype=np.int32)
        tw_feat_idx = np.array(tw_feat_idx, dtype=np.int64).T

        feat_num = len(feat_to_idx)
        tw_feat_mat = sp.sparse.coo_matrix(
            (tw_feat_val, (tw_feat_idx[0, :], tw_feat_idx[1, :])),
            shape=(tw_num, feat_num),
            dtype=np.int8)

        tw_adj = tw_feat_mat @ tw_feat_mat.T
        return tw_adj

    def tweet_to_event(self, data):
        ev_ids = sorted(set(it.event_id for it in data))
        ev_to_idx = {eid: i for i, eid in enumerate(ev_ids)}
        tw_to_ev = [ev_to_idx[it.event_id] for it in data]
        return tw_to_ev, ev_to_idx

    def build_feats_adj(self, data, feats):
        feats_adj = [func(data) for f, func in feats]
        return feats_adj

    def build_feat_adj(self, data, cols):
        tw_num = len(data)
        tw_feat_idx = []
        feat_to_idx = {}
        COLUMNS = [
            'tweet_id', 'text', 'event_id', 'words', 'filtered_words',
            'entities', 'user_id', 'created_at', 'urls', 'hashtags', 'user_mentions'
        ]
        DataItem = namedtuple('DataItem', COLUMNS)
        cols = [DataItem._fields.index(c) for c in cols] if isinstance(cols, list) else [DataItem._fields.index(cols)]
        for i, it in enumerate(data):
            feats = [
                list(itertools.chain(*it[c])) if isinstance(it[c], list) or isinstance(it[c], tuple) else [it[c]]
                for c in cols  # 特征列
            ]
            feats = [f for cf in feats for f in cf]

            for f in feats:
                if f not in feat_to_idx:
                    feat_to_idx[f] = len(feat_to_idx)
                f_idx = feat_to_idx[f]

                tw_feat_idx.append([i, f_idx])

        tw_feat_val = np.ones((len(tw_feat_idx),), dtype=np.int32)
        tw_feat_idx = np.array(tw_feat_idx, dtype=np.int64).T

        feat_num = len(feat_to_idx)
        tw_feat_mat = sp.sparse.coo_matrix(
            (tw_feat_val, (tw_feat_idx[0, :], tw_feat_idx[1, :])),
            shape=(tw_num, feat_num),
            dtype=np.int8)

        tw_adj = tw_feat_mat @ tw_feat_mat.T
        return tw_adj

    def get_time_relation(self, tw_i, tw_j, delta: datetime.timedelta = datetime.timedelta(hours=4)):
        a, b = tw_i.created_at, tw_j.created_at
        return int(a - b < delta) if a > b else int(b - a < delta)

    def make_train_samples(self, tw_adj, tw_to_ev, data):
        tw_adj_num = len(tw_adj)
        tw_num = len(tw_to_ev)
        ev_num = max(tw_to_ev) + 1

        tw_ev_mat = np.zeros(shape=(tw_num, ev_num), dtype=np.int8)
        for i, e in enumerate(tw_to_ev):
            tw_ev_mat[i, e] = 1

        eye = sp.sparse.eye(tw_num, tw_num, dtype=np.int8)
        adj = tw_adj[0] - eye
        for f in range(1, tw_adj_num):
            adj = adj + (tw_adj[f] - eye)

        adj = np.asarray(adj.todense())

        pairs = []
        for i in range(tw_num):
            ev_idx = tw_to_ev[i]
            ev_tw_vec = tw_ev_mat[:, ev_idx]
            ev_tw_num = ev_tw_vec.sum()
            if ev_tw_num < 5:
                # print(f"outlier or small events: {i} -- {tw_to_ev[i]}--{ev_tw_num[tw_to_ev[i]]}")
                continue

            adj_i_tw = adj[i, :]
            adj_i_tw_score = np.exp(adj_i_tw - (1. - ev_tw_vec) * 1e12)

            pos_idx, = np.nonzero(ev_tw_vec)
            p = sp.special.softmax(adj_i_tw_score.take(pos_idx))

            pos_idx = np.random.choice(pos_idx, size=args.SAMPLE_NUM_TWEET, p=p)
            # (tag, event, (tweet_a, tweet_b), [feats,])
            pos_pairs = [
                (
                    int(tw_to_ev[i] == tw_to_ev[j]), tw_to_ev[i], (i, j),
                    list(1 if tw_adj[f][i, j] > 0 else 0 for f in range(tw_adj_num)) + [
                        self.get_time_relation(data[i], data[j])]
                )
                for j in pos_idx
            ]
            pairs.extend(pos_pairs)

            neg_idx, = np.nonzero(1 - ev_tw_vec)
            adj_i_tw_score = np.exp(adj_i_tw - ev_tw_vec * 1e12)

            p = sp.special.softmax(adj_i_tw_score.take(neg_idx))

            neg_idx = np.random.choice(neg_idx, size=args.SAMPLE_NUM_TWEET, p=p)

            # (tag, event, (tweet_a, tweet_b), [feats,])
            neg_pairs = [
                (
                    int(tw_to_ev[i] == tw_to_ev[j]), tw_to_ev[i], (i, j),
                    list(1 if tw_adj[f][i, j] > 0 else 0 for f in range(tw_adj_num)) + [
                        self.get_time_relation(data[i], data[j])]
                )
                for j in neg_idx
            ]
            pairs.extend(neg_pairs)

        return pairs

    def make_ref_samples(self, tw_adj, tw_to_ev, data):
        tw_adj_num = len(tw_adj)
        tw_num = len(tw_to_ev)

        pairs = []
        adj = tw_adj[0]
        for f in range(1, tw_adj_num):
            adj = adj + tw_adj[f]

        adj = np.asarray(adj.todense())
        eye = np.eye(tw_num, tw_num, dtype=np.int8)
        adj = adj * (1 - eye) + eye

        tw_idx = np.arange(tw_num)
        for i in range(tw_num):
            p = sp.special.softmax(np.exp(adj[i]))

            ref_idx = np.random.choice(tw_idx, size=args.SAMPLE_NUM_TWEET * 3, p=p)
            # (tag, event, (tweet_a, tweet_b), [feats,])
            ref_pairs = [
                (
                    int(tw_to_ev[i] == tw_to_ev[j]),
                    tw_to_ev[i], (i, j),
                    list(1 if tw_adj[f][i, j] > 0 else 0 for f in range(tw_adj_num)) + [
                        self.get_time_relation(data[i], data[j])]
                )
                for j in ref_idx
            ]
            pairs.extend(ref_pairs)

        return pairs

    def process_block(self, block):
        blk = {}

        FEAT_COLS = [
            ("entities", self.build_entity_adj),
            ("hashtags", self.build_hashtag_adj),
            ("user", self.build_user_adj),  # user_mentions and user_id
            ("words", self.build_words_adj),

            # ("create_at", self.build_creat_at_adj)
        ]

        for name in ["train", "test", "valid"]:
            data = block[name]
            tw_to_ev, ev_to_idx = self.tweet_to_event(data)
            tw_adj = self.build_feats_adj(data, FEAT_COLS)

            blk[name] = {
                "data": data,
                "tw_to_ev": tw_to_ev,
                "ev_to_idx": ev_to_idx,
                "tw_adj": tw_adj
            }

            if name == "train" or name == "valid":
                if data:
                    blk[name]["samples"] = self.make_train_samples(tw_adj, tw_to_ev, data)
                else:
                    blk[name]["samples"] = []

            if name == "test":
                if data:
                    blk[name]["samples"] = self.make_ref_samples(tw_adj, tw_to_ev, data)
                else:
                    blk[name]["samples"] = []

        return blk

    def split_train_test_validation(self, data: List):
        block = []
        off_dataset = []
        for i in range(len(data)):
            if i == 0:
                data_size = len(data[i])
                valid_size = math.ceil(data_size * 0.2)
                test_size = math.ceil(data_size * 0.1)  # Add test size
                train, temp = train_test_split(data[i], test_size=valid_size + test_size, random_state=42, shuffle=True)
                valid, test = train_test_split(temp, test_size=test_size, random_state=42, shuffle=True)
                block.append({"train": train, "test": test, "valid": valid})

                print(f"Block {i}: Train size: {len(train)}, Valid size: {len(valid)}, Test size: {len(test)}")

                off_test_size = math.ceil(data_size * 0.2)
                off_valid_size = math.ceil(data_size * 0.1)
                off_train, off_test = train_test_split(data[i], test_size=off_test_size, random_state=42, shuffle=True)
                off_train, off_valid = train_test_split(off_train, test_size=off_valid_size, random_state=42,
                                                        shuffle=True)

                print("create offline dataset ...", end="\t")
                off_dataset.append(self.process_block({"train": off_train, "test": off_test, "valid": off_valid}))
                print("done")

                print(f"save data to '{args.file_path}cache/offline.npy' ... ", end='')
                np.save(args.file_path + 'cache/offline.npy', off_dataset)
                print("\tDone")

            elif i % args.WINDOW_SIZE == 0:
                sub_data = []
                for j in range(args.WINDOW_SIZE):
                    sub_data += data[i - j]
                sub_data_size = len(sub_data)
                sub_valid_size = math.ceil(sub_data_size * 0.2)
                train, valid = train_test_split(sub_data, test_size=sub_valid_size, random_state=42, shuffle=True)
                block.append({"train": train, "test": data[i], "valid": valid})
                print(f"Block {i}: Train size: {len(train)}, Valid size: {len(valid)}, Test size: {len(data[i])}")
            else:
                block.append({"train": [], "test": data[i], "valid": []})
                print(f"Block {i}: Train size: 0, Valid size: 0, Test size: {len(data[i])}")

        return block

    def split_into_blocks(self, data):
        COLUMNS = [
            'tweet_id', 'text', 'event_id', 'words', 'filtered_words',
            'entities', 'user_id', 'created_at', 'urls', 'hashtags', 'user_mentions'
        ]
        DataItem = namedtuple('DataItem', COLUMNS)
        data = [DataItem(*it) for it in data]
        data = sorted(data, key=lambda it: it.created_at)
        groups = itertools.groupby(data, key=lambda it: it.created_at.timetuple().tm_yday)
        groups = {k: list(g) for k, g in groups}

        days = sorted(groups.keys())
        blk0 = [groups[d] for d in days[:7]]
        blk0 = [it for b in blk0 for it in b]

        print(f"Initial Block 0: {len(blk0)} items")

        day_blk = [groups[d] for d in days[7:-1]]
        for idx, blk in enumerate(day_blk, start=1):
            print(f"Block {idx}: {len(blk)} items")

        blocks = [blk0] + day_blk
        datacount = [len(sublist) for sublist in blocks]

        print(f"save block datas counts into '{args.file_path}cache/datacount.npy' ", end='')
        os.makedirs(f'{args.file_path}cache', exist_ok=True)
        np.save(f'{args.file_path}cache/datacount.npy', datacount)
        print("done")

        return self.split_train_test_validation(blocks)

    def pre_process(self, data):
        print("split data into blocks... ")
        blocks = self.split_into_blocks(data)
        print("\tDone")

        print("process blocks..., ", end='')
        data_blocks = []
        for i, blk in enumerate(blocks):
            print(i, end=" ")
            blk = self.process_block(blk)
            data_blocks.append(blk)

        print("\tDone")
        return data_blocks


class RPLMSED:
    """RPLMSED (Representation Learning-based Pre-trained Language Model for Social Event Detection) class.
    
    This class implements event detection using pre-trained language models and representation learning.
    """
    
    def __init__(self, args, dataset):
        """Initialize RPLMSED model.
        
        Args:
            args: Configuration arguments
            dataset: Input dataset
        """
        self.dataset = dataset
        self.args = args
        self.model = None

    def preprocess(self):
        """Preprocess the input dataset."""
        preprocessor = Preprocessor()
        preprocessor.preprocess(self.dataset)

    def fit(self):
        """Train the model."""
        torch.manual_seed(2357)
        self.args.model_name = os.path.basename(os.path.normpath(self.args.plm_path))
        dataset_name = os.path.basename(self.args.dataset)
        self.args.dataset_name = dataset_name.replace(".npy", "")

        if 'cuda' in self.args.device:
            torch.cuda.manual_seed(2357)

        tokenizer = AutoTokenizer.from_pretrained(self.args.plm_path)
        data_blocks = load_data_blocks(self.args.dataset, self.args, tokenizer)
        self.model = start_run(self.args, data_blocks)

    def detection(self):
        """Perform event detection on test data.
        
        Returns:
            tuple: (predictions, ground_truths)
        """
        blk = torch.load(f'{self.args.file_path}cache/cache_long_tail/roberta-large-twitter12.npy')
        test = blk['test']

        msg_tags = np.array(test['tw_to_ev'], dtype=np.int32)
        tst_num = msg_tags.shape[0]
        msg_feats = torch.zeros((tst_num, self.model.feat_size()), device='cpu')
        ref_num = torch.zeros((tst_num,), dtype=torch.long, device='cpu')

        msg_feats = msg_feats / (ref_num + torch.eq(ref_num, 0).float()).unsqueeze(-1)
        msg_feats = msg_feats.numpy()

        n_clust = len(test['ev_to_idx'])
        k_means_score = run_kmeans(msg_feats, n_clust, msg_tags)

        k_means = KMeans(init="k-means++", n_clusters=n_clust, n_init=40, random_state=0)
        k_means.fit(msg_feats)

        predictions = k_means.labels_
        ground_truths = msg_tags
        return predictions, ground_truths

    def evaluate(self, predictions, ground_truths):
        """Evaluate detection results.
        
        Args:
            predictions: Model predictions
            ground_truths: Ground truth labels
            
        Returns:
            tuple: (ars, ami, nmi) evaluation metrics
        """
        ars = metrics.adjusted_rand_score(ground_truths, predictions)

        # Calculate Adjusted Mutual Information (AMI)
        ami = metrics.adjusted_mutual_info_score(ground_truths, predictions)

        # Calculate Normalized Mutual Information (NMI)
        nmi = metrics.normalized_mutual_info_score(ground_truths, predictions)

        print(f"Model Adjusted Rand Index (ARI): {ars}")
        print(f"Model Adjusted Mutual Information (AMI): {ami}")
        print(f"Model Normalized Mutual Information (NMI): {nmi}")
        return ars, ami, nmi


def get_model(args):
    return PairPfxTuningEncoder(
        len(args.pmt_feats), args.plm_path, args.plm_tuning,
        use_ctx_att=args.use_ctx_att, ctx_att_head_num=args.ctx_att_head_num)


def initialize(model, args, num_train_batch):
    # parameters = model.parameters()  # 优化器的初始化
    parameters = [
        {
            'name': 'pair_cls',
            'params': model.pair_cls.parameters(),
            'lr': args.lr
        }, {
            'name': 'pfx_embedding',
            'params': model.pfx_embedding.parameters(),
            'lr': args.lr
        }
    ]

    if args.plm_tuning:
        parameters.append(
            {
                'name': 'encoder',
                'params': model.plm.parameters(),
                'lr': args.lr / 100.
            }
        )

    if args.optimizer == 'Adam':
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'RAdam':
        optimizer = optim.RAdam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise Exception("unsupported optimizer %s" % args.optimizer)

    lr_scheduler = None
    if args.step_lr_gamma > 0:
        lr_scheduler = StepLR(optimizer, step_size=num_train_batch, gamma=args.step_lr_gamma)

    return optimizer, lr_scheduler


def batch_to_tensor(batch, args):
    tags = [tag for tag, evt, a, b, fix, tok, _ in batch]
    events = [evt for tag, evt, a, b, fix, tok, _ in batch]
    prefix = [fix for tag, evt, a, b, fix, tok, _ in batch]
    toks = [tok for tag, evt, a, b, fix, tok, _ in batch]
    typs = [typ for tag, evt, a, b, fix, tok, typ in batch]

    max_len = min(max([len(it) for it in toks]), 200)
    toks = [pad_seq(it, pad=args.pad_tok_id, max_len=max_len) for it in toks]
    toks = torch.tensor(toks, dtype=torch.long)
    typs = [pad_seq(it, pad=args.pad_tok_id, max_len=max_len) for it in typs]
    typs = torch.tensor(typs, dtype=torch.long)
    tags = torch.tensor(tags, dtype=torch.long)
    events = torch.tensor(events, dtype=torch.long)
    prefix = torch.tensor(prefix, dtype=torch.long)

    return toks, typs, prefix, tags, events


# loss functions
# cls_loss = torch.nn.BCEWithLogitsLoss()

def create_trainer(model, optimizer, lr_scheduler, args):
    evt_proto = torch.zeros((args.train_evt_num, model.feat_size()), device=args.device, requires_grad=False)
    cls_loss = torch.nn.BCEWithLogitsLoss()

    # update event cluster center prototype by training batch
    def update_evt_proto(feats, events, alpha):
        proto = torch.zeros_like(evt_proto)
        proto.index_add_(0, events, feats)

        ev_idx, ev_idx_inv, ev_count = torch.unique(events, return_inverse=True, return_counts=True)
        proto_a = torch.index_select(proto, dim=0, index=ev_idx) / ev_count.unsqueeze(-1)
        proto_b = torch.index_select(evt_proto, dim=0, index=ev_idx)

        source = alpha * proto_a + (1 - alpha) * proto_b
        # source = proto_a
        source.detach_()
        source.requires_grad = False

        evt_proto.index_copy_(0, ev_idx, source)

        return proto_a

    # training logic for iteration
    def _train_step(_, batch):
        data = batch_to_tensor(batch, args)
        dist_loss = torch.nn.PairwiseDistance()

        toks, typs, prefix, tags, events = [x.to(args.device) for x in data]
        mask = torch.not_equal(toks, args.pad_tok_id).to(args.device)

        model.train()
        logit, left_feat = model(toks, typs, prefix, mask)

        loss = cls_loss(logit, tags.float())
        pred = torch.gt(logit, 0.)

        feats = left_feat
        evt_feats = update_evt_proto(feats, events, 0.8)
        protos = evt_proto.index_select(0, events)

        intra_dist = dist_loss(feats, protos)
        intra_dist_loss = intra_dist.mean()

        rand_idx = torch.randperm(evt_feats.size(0), device=args.device)
        rand_evt_feats = evt_feats.index_select(0, rand_idx)
        inter_dist_loss = torch.nn.functional.pairwise_distance(evt_feats, rand_evt_feats)

        inter_dist_loss = torch.maximum(10 - inter_dist_loss, torch.zeros_like(inter_dist_loss)).mean()

        if args.lmda1 > 0.:
            loss = loss + args.lmda1 * inter_dist_loss + args.lmda2 * intra_dist_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()

        del toks, prefix, mask
        acc = accuracy_score(tags.cpu(), pred.cpu())

        return loss, acc, inter_dist_loss, intra_dist_loss

    # Define trainer engine
    trainer = Engine(_train_step)

    # metrics for trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'acc')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'inter')
    RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'intra')

    # Add progress bar showing trainer metrics
    mtcs = ['loss', 'acc', 'inter', 'intra']
    ProgressBar(persist=True).attach(trainer, mtcs)
    return trainer


def create_evaluator(model, args):
    cls_loss = torch.nn.BCEWithLogitsLoss()

    def _validation_step(_, batch):
        model.eval()
        with torch.no_grad():
            data = batch_to_tensor(batch, args)

            toks, typs, prefix, tags, events = [x.to(args.device) for x in data]
            mask = torch.not_equal(toks, args.pad_tok_id).to(args.device)

            logit, left_feat = model(toks, typs, prefix, mask)

            loss = cls_loss(logit, tags.float())
            pred = torch.gt(logit, 0.)

            acc = accuracy_score(tags.cpu(), pred.cpu())

            return loss, acc

    evaluator = Engine(_validation_step)
    Average(lambda x: x[0]).attach(evaluator, 'loss')
    Average(lambda x: x[1]).attach(evaluator, 'acc')

    ProgressBar(persist=True).attach(evaluator)
    return evaluator


def create_tester(model, args, msg_feats, ref_num):
    cls_loss = torch.nn.BCEWithLogitsLoss()

    def _test_step(_, batch):
        model.eval()
        with torch.no_grad():
            data = batch_to_tensor(batch, args)

            toks, typs, prefix, tags, events = [x.to(args.device) for x in data]
            mask = torch.not_equal(toks, args.pad_tok_id).to(args.device)

            idx = [a for tag, evt, a, b, fix, tok, _ in batch]
            idx = torch.tensor(idx, dtype=torch.long).to(args.device)

            me = [True if a == b else False for tag, evt, a, b, fix, tok, _ in batch]
            me = torch.tensor(me, dtype=torch.long).to(args.device)

            logit, left_feat = model(toks, typs, prefix, mask)

            loss = cls_loss(logit, tags.float())
            pred = torch.gt(logit, 0.)

            # top_k_values, top_k_indices = torch.topk(torch.sigmoid(logit), k=90, largest=True)#

            msk = torch.gt(torch.sigmoid(logit), args.tao)

            acc = accuracy_score(tags.cpu(), pred.cpu())
            msk = torch.logical_or(msk, me)

            msk_idx, = torch.nonzero(msk, as_tuple=True)
            idx = torch.index_select(idx, dim=0, index=msk_idx)
            # idx = torch.index_select(idx, dim=0, index=top_k_indices)#
            ## feats = (pfx_feat + left_feat) / 2.
            feats = left_feat
            feat = torch.index_select(feats, dim=0, index=msk_idx)
            evt = torch.index_select(events, dim=0, index=msk_idx)

            # feat = torch.index_select(feats, dim=0, index=top_k_indices)#
            # evt = torch.index_select(events, dim=0, index=top_k_indices)#

            msg_feats.index_add_(0, idx.cpu(), feat.cpu())
            ref_num.index_add_(0, idx.cpu(), torch.ones_like(evt, device='cpu'))

            return loss, acc

    tester = Engine(_test_step)

    Average(lambda x: x[0]).attach(tester, 'loss')
    Average(lambda x: x[1]).attach(tester, 'acc')

    ProgressBar(persist=True).attach(tester)
    return tester


def test_on_block(model, cfg, blk, b=0):
    test = blk['test']
    print("Length of test['samples']:", len(test['samples']))

    msg_tags = np.array(test['tw_to_ev'], dtype=np.int32)

    tst_num = msg_tags.shape[0]
    msg_feats = torch.zeros((tst_num, model.feat_size()), device='cpu')  # cfg.feat_dim
    ref_num = torch.zeros((tst_num,), dtype=torch.long, device='cpu')

    train, valid = blk['train'], blk['valid']
    cfg.train_evt_num = len(train['ev_to_idx'])
    # print("cfg.train_evt_num:", cfg.train_evt_num)
    test_gen, test_batch_num = data_generator(test['samples'], cfg.batch_size)
    tester = create_tester(model, cfg, msg_feats, ref_num)

    print("Evaluate model on test data ...")
    test_state = tester.run(test_gen, epoch_length=test_batch_num)

    print("Available metrics:", test_state.metrics.keys())  # Add debug print to check available metrics
    test_metrics = [(m, test_state.metrics[m]) for m in ['loss', 'acc']]
    test_metrics = ", ".join([("%s: %.4f" % (m, v)) for m, v in test_metrics])
    print(f"Test: {test_metrics}\n", flush=True)

    # clustering & report
    msg_feats = msg_feats / (ref_num + torch.eq(ref_num, 0).float()).unsqueeze(-1)
    msg_feats = msg_feats.numpy()
    n_clust = len(test['ev_to_idx'])

    if not os.path.exists(cfg.eva_data):
        os.makedirs(cfg.eva_data)

    Evaluate_datas = {'msg_feats': msg_feats, 'msg_tags': msg_tags, 'n_clust': n_clust}
    if cfg.offline:
        print(f"save Evaluate_datas_offline to '{cfg.eva_data}/evaluate_data_long_tail.npy'", end='\t')
        np.save(f'{cfg.eva_data}/evaluate_data_long_tail.npy', Evaluate_datas)
    else:
        print(f"save Evaluate_datas{b} to '{cfg.eva_data}/evaluate_data_M{b}.npy'", end='\t')
        e_path = cfg.eva_data + f'/evaluate_data_M{b}.npy'
        np.save(e_path, Evaluate_datas)

    print('done')

    k_means_score = run_kmeans(msg_feats, n_clust, msg_tags)
    dbscan_score = run_hdbscan(msg_feats, msg_tags)

    del msg_feats

    return k_means_score, dbscan_score


def load_ckpt(model, args, ckpt, b):
    print(f"Load best ckpt for block {b} from '{ckpt}'")

    ckpt = torch.load(ckpt, map_location=args.device)
    model.load_state_dict(ckpt['model'], strict=False)

    ckpt_args = ckpt['args']
    ckpt_args.dataset = args.dataset
    ckpt_args.plm_path = args.plm_path
    ckpt_args.batch_size = args.batch_size
    ckpt_args.device = args.device
    ckpt_args.tao = args.tao

    return model, ckpt_args


def start_run(cfg, blocks):
    tokenizer = AutoTokenizer.from_pretrained(args.plm_path)
    cfg.pad_tok_id = tokenizer.pad_token_id
    model = get_model(cfg).to(cfg.device)
    # print settings
    print_table([(k, str(v)[0:60]) for k, v in vars(cfg).items()])
    kmeans_scores, dbscan_scores = [], []
    ckpt_list = []

    for b, blk in enumerate(blocks):
        print(f"Processing block-{b}...", flush=True)
        print(f"Block-{b} content keys: {blk.keys()}")

        train, valid, test = (blk[n] for n in ('train', 'valid', 'test'))
        print(
            f"Train samples: {len(train['samples'])}, Valid samples: {len(valid['samples'])}, Test samples: {len(test['samples'])}")

        if b > 0:
            print(f"test model on data block-{b} ...", flush=True)
            kms_score, dbs_score = test_on_block(model, cfg, blk, b)
            kmeans_scores.append(kms_score)
            dbscan_scores.append(dbs_score)

            print("KMeans:")
            print_scores(kmeans_scores)
            print("DBSCAN:")
            print_scores(dbscan_scores)

        if b % 3 == 0:
            gc.collect()
            print(f"train on data block-{b} ...", flush=True)
            model, ckpt = train_on_block(model, cfg, blk, b)
            ckpt_list.append(ckpt)

        if b == 0 and args.offline:
            print(f"close test on data block-{b} ...", flush=True)
            kms_score, dbs_score = test_on_block(model, args, blk, b)
            kmeans_scores.append(kms_score)
            dbscan_scores.append(dbs_score)

            print("KMeans:")
            print_scores(kmeans_scores)
            print("DBSCAN:")
            print_scores(dbscan_scores)

    if args.offline:
        ckpt_list_file = os.path.join(args.ckpt_path, 'best_off_model.txt')
    else:
        ckpt_list_file = os.path.join(args.ckpt_path, 'ckpt_list.txt')

    with open(ckpt_list_file, 'w', encoding='utf8') as f:
        ckpt_list = [str(p) for p in ckpt_list]
        f.write("\n".join(ckpt_list))

    return model


def train_on_block(model, args, blk, blk_id=0):
    # reload plm in tuning mode
    if blk_id > 0 and args.plm_tuning:
        print("accumulate reload PLM parameters", flush=True)
        model.accumulate_reload_plm(args.device)
    train, valid = blk['train'], blk['valid']
    # fewer data for code test
    ###
    # train['samples'] = train['samples'][:500]
    # valid['samples'] = valid['samples'][:200]

    args.train_evt_num = len(train['ev_to_idx'])

    train_gen, train_batch_num = data_generator(train['samples'], args.batch_size, True, True)
    valid_gen, valid_batch_num = data_generator(valid['samples'], args.batch_size, False, True)

    # create model, optimizer and learning rate scheduler
    optimizer, lr_scheduler = initialize(model, args, train_batch_num)

    # print model parameters
    # summary(model, input_size=((args.batch_size, 50), (args.batch_size, 50)))

    # Setup model trainer and evaluator
    trainer = create_trainer(model, optimizer, lr_scheduler, args)
    evaluator = create_evaluator(model, args)

    @trainer.on(Events.EPOCH_STARTED)
    def log_learning_rates(_):
        for param_group in optimizer.param_groups:
            print("{} lr: {:>1.2e}".format(param_group.get('name', ''), param_group["lr"]))

    # Run model evaluation every epoch and show results
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def evaluate_model():  # eng
        print("Evaluate model on eval data ...")
        val_state = evaluator.run(valid_gen, epoch_length=valid_batch_num)

        eval_metrics = [(m, val_state.metrics[m]) for m in ['loss', 'acc']]
        eval_metrics = ", ".join([("%s: %.4f" % (m, v)) for m, v in eval_metrics])

        print(f"Eval: {eval_metrics}\n", flush=True)

    def score_function(_):
        if args.early_stop_monitor == 'loss':
            return - evaluator.state.metrics['loss']
        elif args.early_stop_monitor in evaluator.state.metrics:
            return evaluator.state.metrics[args.early_stop_monitor]
        else:
            raise Exception('unsupported metric %s' % args.early_stop_monitor)

    if args.offline:
        filename_prefix = f"{args.model_name}-{'tuning' if args.plm_tuning else 'fixed'}-{args.dataset_name}-offline"
    else:
        filename_prefix = f"{args.model_name}-{'tuning' if args.plm_tuning else 'fixed'}-{args.dataset_name}-{blk_id}"
    ckpt_handler = ModelCheckpoint(args.ckpt_path, score_function=score_function,
                                   filename_prefix=filename_prefix, n_saved=3,
                                   create_dir=True, require_empty=False)

    # if not tuning plm,
    model_state = get_model_state(model, ['pair_cls', 'pfx_embedding'], args.plm_tuning)
    ckpt = {'model': model_state, 'args': CkptWrapper(args)}
    trainer.add_event_handler(Events.EPOCH_COMPLETED(every=1), ckpt_handler, ckpt)

    hdl_early_stop = EarlyStopping(patience=args.early_stop_patience, score_function=score_function, trainer=trainer)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    evaluator.add_event_handler(Events.COMPLETED, hdl_early_stop)

    # start training loop
    trainer.run(train_gen, max_epochs=args.max_epochs, epoch_length=train_batch_num)

    # load best checkpoint
    best_ckpt = ckpt_handler.last_checkpoint
    print(f"Load best model checkpoint from '{best_ckpt}'")
    ckpt = torch.load(best_ckpt)
    model.load_state_dict(ckpt['model'], strict=False)
    del ckpt
    return model, best_ckpt


# utils

def load_data_blocks(path_to_data, args, tokenizer):
    print(f"load data from '{path_to_data}'... ", end='')
    dataset = np.load(path_to_data, allow_pickle=True)
    print("\tDone")

    path_to_blocks = []
    print(f"encode block samples, ")

    for i, blk in enumerate(dataset):
        print(f"Message Block{i}", flush=True)
        train, valid, test = (blk[n] for n in ('train', 'valid', 'test'))
        print(
            f"Train samples: {len(train['samples'])}, Valid samples: {len(valid['samples'])}, Test samples: {len(test['samples'])}")

        path = f"{args.file_path}/cache/cache_long_tail/"

        if not os.path.exists(path):
            os.makedirs(path)
        if args.offline:
            # blk_path = os.path.join(path, f"{args.model_name}-{args.dataset_name}-offline.npy")
            blk_path = os.path.join(path, f"{args.model_name}-{args.dataset_name}.npy")
        else:
            blk_path = os.path.join(path, f"{args.model_name}-{args.dataset_name}-M{i + 1}.npy")

        if not os.path.exists(blk_path):
            print("train dateset processing", end=" ")
            train['samples'] = encode_samples(train['samples'], train['data'], tokenizer, args.pmt_feats)
            print("done")

            print("valid dateset processing", end=" ")
            valid['samples'] = encode_samples(valid['samples'], valid['data'], tokenizer, args.pmt_feats)
            print("done")

            print("test dateset processing", end=" ")
            test['samples'] = encode_samples(test['samples'], test['data'], tokenizer, args.pmt_feats)
            print("done")

            torch.save(
                {'train': train, 'valid': valid, 'test': test},
                blk_path
            )
        if blk_path not in path_to_blocks:
            path_to_blocks.append(blk_path)

    del dataset
    print("Done")

    path_to_blocks = ['../model/model_saved/rplmsed/cache/cache_long_tail/roberta-large-twitter12.npy']
    for blk_path in path_to_blocks:
        print(f"load block from '{blk_path}'... \n", end='')
        loaded_blk = torch.load(blk_path)
        # 检查加载的数据是否为字典
        if isinstance(loaded_blk, dict):
            print(f"Loaded block type: {type(loaded_blk)}")
            print(f"Loaded block keys: {loaded_blk.keys()}")
            yield loaded_blk
        else:
            print(f"Error: Loaded block is not a dictionary, but {type(loaded_blk)}")
            yield None


class CkptWrapper:
    def __init__(self, state: Any):
        self.state = state

    def state_dict(self):
        return self.state


def get_model_state(model, params, plm_tuning):
    if plm_tuning:
        return model
    else:
        model_state = model.state_dict()
        state = OrderedDict()
        for k, v in model_state.items():
            for p in params:
                if k.startswith(p):
                    state[k] = model_state[k]
                    break

        return CkptWrapper(state)


def width(text):
    return sum([2 if '\u4E00' <= c <= '\u9FA5' else 1 for c in text])


def print_table(tab):
    col_width = [max(width(x) for x in col) for col in zip(*tab)]
    print("+-" + "-+-".join("{:-^{}}".format('-', col_width[i]) for i, x in enumerate(tab[0])) + "-+")
    for line in tab:
        print("| " + " | ".join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)) + " |")
    print("+-" + "-+-".join("{:-^{}}".format('-', col_width[i]) for i, x in enumerate(tab[0])) + "-+")


def data_generator(data, batch_size, shuffle=False, repeat=False):
    batch_num = math.ceil(len(data) / batch_size)
    return create_data_generator(data, batch_size, shuffle, repeat, batch_num), batch_num


def create_data_generator(data, batch_size, shuffle, repeat, batch_num):
    while True:
        if shuffle:
            shuffled_idx = [i for i in range(len(data))]
            random.shuffle(shuffled_idx)
            data = [data[i] for i in shuffled_idx]

        batch_id = 0
        while batch_id < batch_num:
            offset = batch_id * batch_size
            batch_data = data[offset:offset + batch_size]
            yield batch_data

            batch_id = batch_id + 1
        if repeat:
            continue
        else:
            break


def pad_seq(seq, max_len, pad=0, pad_left=False):
    """
    padding or truncate sequence to fixed length
    :param seq: input sequence
    :param max_len: max length
    :param pad: padding token id
    :param pad_left: pad on left
    :return: padded sequence
    """
    if max_len < len(seq):
        seq = seq[:max_len]
    elif max_len > len(seq):
        padding = [pad] * (max_len - len(seq))
        if pad_left:
            seq = padding + seq
        else:
            seq = seq + padding
    return seq


def run_kmeans(msg_feats, n_clust, msg_tags):
    # defalut:10
    k_means = KMeans(init="k-means++", n_clusters=n_clust, n_init=40, random_state=0)
    k_means.fit(msg_feats)

    msg_pred = k_means.labels_
    score_funcs = [
        ("NMI", metrics.normalized_mutual_info_score),
        ("AMI", metrics.adjusted_mutual_info_score),
        ("ARI", metrics.adjusted_rand_score),
    ]

    scores = {m: fun(msg_tags, msg_pred) for m, fun in score_funcs}

    return scores


def run_hdbscan(msg_feats, msg_tags):
    hdb = HDBSCAN(min_cluster_size=8)
    hdb.fit(msg_feats)

    msg_pred = hdb.labels_
    score_funcs = [
        ("NMI", metrics.normalized_mutual_info_score),
        ("AMI", metrics.adjusted_mutual_info_score),
        ("ARI", metrics.adjusted_rand_score),
    ]

    scores = {m: fun(msg_tags, msg_pred) for m, fun in score_funcs}

    return scores


def run_dbscan(msg_feats, msg_tags):
    db = OPTICS(min_cluster_size=8, xi=0.01)
    db.fit(msg_feats)

    msg_pred = db.labels_
    score_funcs = [
        ("NMI", metrics.normalized_mutual_info_score),
        ("AMI", metrics.adjusted_mutual_info_score),
        ("ARI", metrics.adjusted_rand_score),
    ]

    scores = {m: fun(msg_tags, msg_pred) for m, fun in score_funcs}

    return scores


def print_scores(scores):
    line = [' ' * 4] + [f'   M{i:02d} ' for i in range(1, len(scores) + 1)]
    print("".join(line))

    score_names = ['NMI', 'AMI', 'ARI']
    for n in score_names:
        line = [f'{n} '] + [f'  {s[n]:1.3f}' for s in scores]
        print("".join(line))
    print('\n', flush=True)


def encode_samples(samples, raw_data, tokenizer, pmt_idx):
    data = []
    for tag, ev_idx, (tw_a, tw_b), pmt_feat in samples:
        tw_a_text = raw_data[tw_a].text
        tw_b_text = raw_data[tw_b].text
        tok = tokenizer(tw_a_text, tw_b_text, padding=True)

        # 只保留需要的关联特征
        # (entities, hashtags, user, words, time)
        pmt_feat = [pmt_feat[f] for f in pmt_idx]

        base = [2 * i for i in range(len(pmt_feat))]
        pmt_ids = [b + f for f, b in zip(pmt_feat, base)]

        if 'token_type_ids' not in tok:
            types = [0, 0, 1, 1]
            token_type_ids = tok.encodings[0].sequence_ids
            j = 0
            for i, t in enumerate(token_type_ids):
                if t is None:
                    token_type_ids[i] = types[j]
                    j += 1
        else:
            token_type_ids = tok['token_type_ids']

        data.append((tag, ev_idx, tw_a, tw_b, pmt_ids, tok['input_ids'], token_type_ids))

    return data


def count_condition(data, key, threshold):
    return sum(entry[key] > threshold for entry in data), sum(entry[key] <= threshold for entry in data)


def calculate_average_min_score(newscore, min_score, max_score):
    for i, score in enumerate(newscore):
        for key, value in score.items():
            min_score[i][key] = min(min_score[i][key], value)
            max_score[i][key] = max(max_score[i][key], value)

    return min_score, max_score


class StructAttention(torch.nn.Module):
    """
    The class is an implementation of the paper A Structured Self-Attentive Sentence Embedding
    """

    def __init__(self, feat_dim, hid_dim, att_head_num=1):
        """
        Initializes parameters suggested in paper
        Args:
            feat_dim:       {int} hidden dimension for lstm
            hid_dim:        {int} hidden dimension for the dense layer
            att_head_num:   {int} attention-hops or attention heads
        Returns:
            self
        Raises:
            Exception
        """
        super(StructAttention, self).__init__()
        self.W1 = torch.nn.Linear(feat_dim, hid_dim, bias=False)
        nn.init.xavier_normal_(self.W1.weight)

        self.W2 = torch.nn.Linear(hid_dim, att_head_num, bias=False)
        nn.init.xavier_normal_(self.W2.weight)

        self.att_head_num = att_head_num

    def forward(self, inpt, mask=None):
        """
        :param inpt: [len, bsz, dim]
        :param mask: [len, bsz]
        :return: [bsz, head_num, dim], [bsz, head_num, len]
        """
        hid = torch.tanh(self.W1(inpt))
        hid = self.W2(hid)

        if mask is not None:
            mask = mask.float().unsqueeze(-1).expand(-1, -1, self.att_head_num)
            mask = (1. - mask) * 1e10
            hid = hid - mask
        att = torch.softmax(hid, dim=0).permute(1, 2, 0)

        outp = att @ inpt.permute(1, 0, 2)

        return outp, att


class PairPfxTuningEncoder(nn.Module):
    def __init__(self, pmt_len,
                 plm_path, plm_tuning=False, from_config=False,
                 use_ctx_att=True, ctx_att_head_num=2):
        super().__init__()
        self.pfx_len = pmt_len
        self.plm_path = plm_path

        if from_config:
            config = AutoConfig.from_pretrained(plm_path)
            self.plm = AutoModel.from_config(config)
        else:
            self.plm = AutoModel.from_pretrained(plm_path)

        if not plm_tuning:
            for name, param in self.plm.named_parameters():
                param.requires_grad = False
                param.detach_()

        self.plm_oupt_dim = self.plm.config.hidden_size

        self.plm_emb_dim = self.plm.embeddings.word_embeddings.embedding_dim

        self.pfx_embedding = nn.Embedding(self.pfx_len * 2, self.plm_emb_dim)
        self.pfx_mask = torch.ones((1, self.pfx_len), dtype=torch.bool)

        self.linear = nn.Linear(self.plm_oupt_dim, self.plm_oupt_dim // 2)

        self.ctx_att = None
        if use_ctx_att:
            self.ctx_att = StructAttention(self.plm_oupt_dim // 2, self.plm_oupt_dim // 4,
                                           att_head_num=ctx_att_head_num)
        self.pair_cls = nn.Linear(2 * (self.plm_oupt_dim // 2), 1)

    def feat_size(self):
        return self.plm_oupt_dim // 2

    def reload_plm(self, device):
        self.plm = AutoModel.from_pretrained(self.plm_path).to(device)

    # 0.4
    def accumulate_reload_plm(self, device, accumulate_rate=0.4):
        origin = AutoModel.from_pretrained(self.plm_path).to('cpu')
        plm_params = self.plm.named_parameters()
        origin_params = origin.named_parameters()
        for ((tgt_name, tgt_param), (src_name, src_param)) in zip(plm_params, origin_params):
            assert (tgt_name == src_name), f"param name {tgt_name} and {src_name} does not match"
            tgt_param.data = (1. - accumulate_rate) * tgt_param.data + accumulate_rate * src_param.to(device).data

    def fix_plm(self):
        for name, param in self.plm.named_parameters():
            param.requires_grad = False
            param.detach_()

    def forward(self, inputs, types, prompt, mask):
        bsz, txt_len = mask.size()

        pmt_msk = self.pfx_mask.to(inputs.device).expand(bsz, -1)
        ext_msk = torch.cat([pmt_msk, mask], dim=-1)
        # ext_msk =mask#

        pmt_emb = self.pfx_embedding(prompt)
        pmt_len = prompt.size(-1)
        txt_emb = self.plm.embeddings(inputs)
        embed = torch.cat([pmt_emb, txt_emb], dim=1)
        # embed= txt_emb #
        att_msk = ext_msk[:, None, None, :]
        att_msk = (1.0 - att_msk.float()) * torch.finfo(torch.float).min

        hidden = self.plm.encoder(embed, att_msk, output_hidden_states=True)['last_hidden_state']
        hidden = torch.tanh(self.linear(hidden))

        pmt_feat = hidden[:, :pmt_len, ...]
        tok_feat = hidden[:, pmt_len:, ...]
        # tok_feat = hidden#

        left_msk = (1 - types) * mask
        left_feat = tok_feat * left_msk.unsqueeze(-1)
        left_msk = torch.cat([pmt_msk.int(), left_msk], dim=1)
        left_feat = torch.cat([pmt_feat, left_feat], dim=1)
        if self.ctx_att is None:
            left_feat = left_feat.sum(dim=-2) / left_msk.sum(-1, keepdims=True)
        else:
            left_feat, left_att = self.ctx_att(left_feat.permute(1, 0, 2), mask=left_msk.permute(1, 0))
            left_feat = torch.mean(left_feat, dim=1)

        right_msk = types * mask
        right_feat = tok_feat * right_msk.unsqueeze(-1)
        if self.ctx_att is None:
            right_feat = right_feat.sum(dim=-2) / right_msk.sum(-1, keepdims=True)
        else:
            right_feat, right_att = self.ctx_att(right_feat.permute(1, 0, 2), mask=right_msk.permute(1, 0))
            right_feat = torch.mean(right_feat, dim=1)

        cls_feat = torch.cat([left_feat, right_feat], dim=-1)

        logit = self.pair_cls(cls_feat).squeeze(dim=-1)

        return logit, left_feat


if __name__ == '__main__':
    from dataset.dataloader import Event2012
    dataset = Event2012().load_data()
    args = args_define().args
    
    rplmsed = RPLMSED(args, dataset)
    rplmsed.preprocess()
    rplmsed.fit()
    predictions, ground_truths = rplmsed.detection()
    rplmsed.evaluate(predictions, ground_truths)
