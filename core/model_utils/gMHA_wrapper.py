import torch.nn as nn
from core.model_utils.gMHA_hadamard import HadamardEncoderLayer
from core.model_utils.gMHA_gt import GTEncoderLayer
from core.model_utils.gMHA_graphormer import GraphormerEncoderLayer
from core.model_utils.mlp_mixer import MixerBlock

class MLPMixer(nn.Module):
    def __init__(self,
                 nhid,
                 nlayer,
                 n_patches,
                 dropout=0,
                 with_final_norm=True
                 ):
        super().__init__()
        self.n_patches = n_patches
        self.with_final_norm = with_final_norm
        self.mixer_blocks = nn.ModuleList(
            [MixerBlock(nhid, self.n_patches, nhid*4, nhid//2, dropout=dropout) for _ in range(nlayer)])
        if self.with_final_norm:
            self.layer_norm = nn.LayerNorm(nhid)

    def forward(self, x, coarsen_adj, mask):
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)
        if self.with_final_norm:
            x = self.layer_norm(x)
        return x


class Hadamard(nn.Module):
    # Hadamard attention (default): (A ⊙ softmax(QK^T/sqrt(d)))V
    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([HadamardEncoderLayer(
            d_model=nhid, dim_feedforward=nhid*2, nhead=nhead, batch_first=batch_first, dropout=dropout)for _ in range(nlayer)])

    def forward(self, x, coarsen_adj, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=coarsen_adj, src_key_padding_mask=mask)
        return x


class Standard(nn.Module):
    # standard (full) attention: softmax(QK^T/sqrt(d))V
    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([GTEncoderLayer(
            d_model=nhid, dim_feedforward=nhid*2, nhead=nhead, batch_first=batch_first, dropout=dropout)for _ in range(nlayer)])

    def forward(self, x, coarsen_adj, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=None, src_key_padding_mask=mask)
        return x


class Graph(nn.Module):
    # Graph attention (GT-like): softmax(A ⊙ QK^T/sqrt(d))V
    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([GTEncoderLayer(
            d_model=nhid, dim_feedforward=nhid*2, nhead=nhead, batch_first=batch_first, dropout=dropout)for _ in range(nlayer)])

    def forward(self, x, coarsen_adj, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=coarsen_adj, src_key_padding_mask=mask)
        return x


class Kernel(nn.Module):
    # Kernel attention (GraphiT-like): softmax(random_walk(A) ⊙ QK^T/sqrt(d))V
    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([GTEncoderLayer(
            d_model=nhid, dim_feedforward=nhid*2, nhead=nhead, batch_first=batch_first, dropout=dropout)for _ in range(nlayer)])

    def forward(self, x, coarsen_adj_dense, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=coarsen_adj_dense, src_key_padding_mask=mask)
        return x


class Addictive(nn.Module):
    # Addictive attention (Graphormer-like): softmax(QK^T/sqrt(d))V + LL(A)
    def __init__(self, nhid, dropout, nlayer, n_patches, nhead=8, batch_first=True):
        super().__init__()
        self.transformer_encoder = nn.ModuleList([GraphormerEncoderLayer(
            d_model=nhid, dim_feedforward=nhid*2, nhead=nhead, batch_first=batch_first, dropout=dropout)for _ in range(nlayer)])

    def forward(self, x, coarsen_adj_dense, mask):
        for layer in self.transformer_encoder:
            x = layer(x, A=coarsen_adj_dense, src_key_padding_mask=mask)
        return x
