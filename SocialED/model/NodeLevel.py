# -*- coding: utf-8 -*-
"""Node-level contrastive learning model for social event detection."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(0)
from torch import optim
# from tensorboardX import SummaryWriter
# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import os
from utils import EMA, set_requires_grad, init_weights, update_moving_average,  currentTime
import copy

from embedder import embedder
from utils import config2string,get_loss
from embedder import Encoder

from process import get_Node_Dataset


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

