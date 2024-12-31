# -*- coding: utf-8 -*-
"""Graph-level contrastive learning model for social event detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)
from torch import optim
# To fix the random seed

from scipy.sparse import csr_matrix
import os
from utils import EMA, set_requires_grad, init_weights, update_moving_average,currentTime,get_loss
import copy
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from embedder import embedder,Encoder
from process import get_Graph_Dataset

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