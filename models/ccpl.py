import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from function import calc_mean_std, nor_mean_std, nor_mean, calc_cov
import random

mlp = nn.ModuleList([nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 16),
                    nn.Linear(128, 128),
                    nn.ReLU(),
                    nn.Linear(128, 32),
                    nn.Linear(256, 256),
                    nn.ReLU(),
                    nn.Linear(256, 64),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 128)]) 

class CCPL(nn.Module):
    def __init__(self, mlp,num_s,tau):
        super(CCPL, self).__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mlp = mlp
        self.num_s=num_s
        self.tau=tau

    def NeighborSample(self, feat, layer, num_s, sample_ids=[]):
        b, c, h, w = feat.size()
        feat_r = feat.permute(0, 2, 3, 1).flatten(1, 2)
        if sample_ids == []:
            dic = {0: -(w+1), 1: -w, 2: -(w-1), 3: -1, 4: 1, 5: w-1, 6: w, 7: w+1}
            s_ids = torch.randperm((h - 2) * (w - 2), device=feat.device) # indices of top left vectors
            s_ids = s_ids[:int(min(num_s, s_ids.shape[0]))]
            ch_ids = (s_ids // (w - 2) + 1) # centors
            cw_ids = (s_ids % (w - 2) + 1)
            c_ids = (ch_ids * w + cw_ids).repeat(8)
            delta = [dic[i // num_s] for i in range(8 * num_s)]
            delta = torch.tensor(delta).to(feat.device)
            n_ids = c_ids + delta
            sample_ids += [c_ids]
            sample_ids += [n_ids]
        else:
            c_ids = sample_ids[0]
            n_ids = sample_ids[1]
        feat_c, feat_n = feat_r[:, c_ids, :], feat_r[:, n_ids, :]
        feat_d = feat_c - feat_n
        for i in range(3):
            feat_d =self.mlp[3*layer+i](feat_d)
        feat_d = Normalize(2)(feat_d.permute(0,2,1))
        return feat_d, sample_ids

    ## PatchNCELoss code from: https://github.com/taesungp/contrastive-unpaired-translation 
    def PatchNCELoss(self, f_q, f_k, tau=0.07):
        # batch size, channel size, and number of sample locations
        B, C, S = f_q.shape
        ###
        f_k = f_k.detach()
        # calculate v * v+: BxSx1
        l_pos = (f_k * f_q).sum(dim=1)[:, :, None]
        # calculate v * v-: BxSxS
        l_neg = torch.bmm(f_q.transpose(1, 2), f_k)
        # The diagonal entries are not negatives. Remove them.
        identity_matrix = torch.eye(S,dtype=torch.bool)[None, :, :].to(f_q.device)
        l_neg.masked_fill_(identity_matrix, -float('inf'))
        # calculate logits: (B)x(S)x(S+1)
        logits = torch.cat((l_pos, l_neg), dim=2) / tau
        # return PatchNCE loss
        predictions = logits.flatten(0, 1)
        targets = torch.zeros(B * S, dtype=torch.long).to(f_q.device)
        return self.cross_entropy_loss(predictions, targets)

    def forward(self,f_q,f_k,layer):
        #e.g. f_k=feats_k[3],layer=3
        f_q, sample_ids = self.NeighborSample(f_q, layer, self.num_s, [])
        f_k, _ = self.NeighborSample(f_k, layer, self.num_s, sample_ids)   
        loss_ccp = self.PatchNCELoss(f_q, f_k, self.tau)
        return loss_ccp

    # def forward(self, feats_q, feats_k,start_layer, end_layer):
    #     loss_ccp = 0.0
    #     for i in range(start_layer, end_layer):
    #         f_q, sample_ids = self.NeighborSample(feats_q[i], i, self.num_s, [])
    #         f_k, _ = self.NeighborSample(feats_k[i], i, self.num_s, sample_ids)   
    #         loss_ccp += self.PatchNCELoss(f_q, f_k, self.tau)
    #     return loss_ccp    
      