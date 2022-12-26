import torch
import torch.nn as nn
from torch_scatter import scatter
from einops.layers.torch import Rearrange


from core.model_utils.elements import MLP
from core.model_utils.feature_encoder import FeatureEncoder
from core.model_utils.gnn import GNN
from core.model_utils.mlp_mixer import MLPMixer
from core.model_utils.transformer import TransformerEncoderLayer


class GraphMLPMixer(nn.Module):

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 node_type, edge_type,
                 gnn_type,
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 use_patch_pe=False):

        super().__init__()
        self.dropout = dropout
        self.n_patches = n_patches
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.use_patch_pe = use_patch_pe
        self.nlayer_gnn = nlayer_gnn
        self.nlayer_mlpmixer = nlayer_mlpmixer
        self.pooling = pooling
        self.res = res
        self.gnn_type = gnn_type

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)

        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                       bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])

        self.reshape = Rearrange('(B p) d ->  B p d', p=self.n_patches)
        self.mlp_mixer = MLPMixer(
            nlayer=nlayer_mlpmixer, nhid=nhid, n_patches=n_patches, dropout=mlpmixer_dropout)

        self.output_decoder = MLP(
            nhid, nout, nlayer=2, with_final_activation=False)

    def forward(self, data):
        # Patch Encoder
        x = self.input_encoder(data.x.squeeze())
        if self.use_rw:
            x += self.rw_encoder(data.rw_pos_enc)
        if self.use_lap:
            x += self.lap_encoder(data.lap_pos_enc)
        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr)

        x = x[data.subgraphs_nodes_mapper]
        e = edge_attr[data.subgraphs_edges_mapper]
        edge_index = data.combined_subgraphs
        batch_x = data.subgraphs_batch
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)

        # MLPMixer
        coarsen_adj = None
        if self.use_patch_pe:
            coarsen_adj = self.reshape(data.coarsen_adj)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        mixer_x = self.reshape(subgraph_x)
        mixer_x = self.mlp_mixer(mixer_x, coarsen_adj)

        # Global Average Pooling
        x = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / \
            data.mask.sum(1, keepdim=True)

        # Readout
        x = self.output_decoder(x)
        return x


class GraphViT(nn.Module):

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 node_type, edge_type,
                 gnn_type,
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 use_patch_pe=False):

        super().__init__()
        self.dropout = dropout
        self.n_patches = n_patches
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.use_patch_pe = use_patch_pe
        self.nlayer_gnn = nlayer_gnn
        self.nlayer_mlpmixer = nlayer_mlpmixer
        self.pooling = pooling
        self.res = res

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)

        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                  bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])

        self.reshape = Rearrange('(B p) d ->  B p d', p=self.n_patches)

        self.transformer_encoder = nn.ModuleList(
            [TransformerEncoderLayer(d_model=nhid, nhead=8, batch_first=True) for _ in range(nlayer_mlpmixer)])

        self.output_decoder = MLP(
            nhid, nout, nlayer=2, with_final_activation=False)

    def forward(self, data):
        # Patch Encoder
        x = self.input_encoder(data.x.squeeze())
        if self.use_rw:
            x += self.rw_encoder(data.rw_pos_enc)
        if self.use_lap:
            x += self.lap_encoder(data.lap_pos_enc)
        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr)

        x = x[data.subgraphs_nodes_mapper]
        e = edge_attr[data.subgraphs_edges_mapper]
        edge_index = data.combined_subgraphs
        batch_x = data.subgraphs_batch
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)

        # MLPMixer
        coarsen_adj = None
        if self.use_patch_pe:
            coarsen_adj = self.reshape(data.coarsen_adj)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        mixer_x = self.reshape(subgraph_x)
        for layer in self.transformer_encoder:
            mixer_x = layer(mixer_x, A_P=coarsen_adj)

        # Global Average Pooling
        x = (mixer_x * data.mask.unsqueeze(-1)).sum(1) / \
            data.mask.sum(1, keepdim=True)

        # Readout
        x = self.output_decoder(x)
        return x


class MPGNN(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 node_type, edge_type,
                 gnn_type,
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean'):

        super().__init__()
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.pooling = pooling

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)

        self.input_encoder = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = GNN(nin=nhid, nout=nhid, nlayer_gnn=nlayer_gnn,
                        gnn_type=gnn_type, bn=bn, dropout=dropout, res=res)

        self.output_decoder = MLP(
            nhid, nout, nlayer=2, with_final_activation=False)

    def forward(self, data):
        x = self.input_encoder(data.x.squeeze())
        if self.use_rw:
            x += self.rw_encoder(data.rw_pos_enc)
        if self.use_lap:
            x += self.lap_encoder(data.lap_pos_enc)

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr)

        x = self.gnns(x, data.edge_index, edge_attr)

        # graph leval task
        x = scatter(x, data.batch, dim=0, reduce=self.pooling)
        x = self.output_decoder(x)

        return x


class GraphMLPMixer4TreeNeighbour(nn.Module):

    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 nlayer_mlpmixer,
                 node_type, edge_type,
                 gnn_type,
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 mlpmixer_dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean',
                 n_patches=32,
                 use_patch_pe=False):

        super().__init__()
        self.dropout = dropout
        self.n_patches = n_patches
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.use_patch_pe = use_patch_pe
        self.nlayer_gnn = nlayer_gnn
        self.nlayer_mlpmixer = nlayer_mlpmixer
        self.pooling = pooling
        self.res = res
        self.gnn_type = gnn_type

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)

        self.layer0_keys = FeatureEncoder(node_type, nfeat_node, nhid)
        self.layer0_values = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = nn.ModuleList([GNN(nin=nhid, nout=nhid, nlayer_gnn=1, gnn_type=gnn_type,
                                       bn=bn, dropout=dropout, res=res) for _ in range(nlayer_gnn)])
        self.U = nn.ModuleList(
            [MLP(nhid, nhid, nlayer=1, with_final_activation=True) for _ in range(nlayer_gnn-1)])

        self.reshape = Rearrange('(B p) d ->  B p d', p=self.n_patches)
        self.reshape2 = Rearrange('B p d -> (B p) d')

        self.mlp_mixer = MLPMixer(
            nlayer=nlayer_mlpmixer, nhid=nhid, n_patches=n_patches, dropout=mlpmixer_dropout)

        self.output_decoder = MLP(
            nhid*2, nout, nlayer=2, with_final_activation=False)

    def forward(self, data):
        # Patch Encoder
        x = data.x
        x_key, x_val = x[:, 0], x[:, 1]
        x_key_embed = self.layer0_keys(x_key)
        x_val_embed = self.layer0_values(x_val)
        x = x_key_embed + x_val_embed

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr)

        x = x[data.subgraphs_nodes_mapper]
        e = edge_attr[data.subgraphs_edges_mapper]
        edge_index = data.combined_subgraphs
        batch_x = data.subgraphs_batch
        for i, gnn in enumerate(self.gnns):
            if i > 0:
                subgraph = scatter(x, batch_x, dim=0,
                                   reduce=self.pooling)[batch_x]
                x = x + self.U[i-1](subgraph)
                x = scatter(x, data.subgraphs_nodes_mapper,
                            dim=0, reduce='mean')[data.subgraphs_nodes_mapper]
            x = gnn(x, edge_index, e)

        # MLPMixer
        coarsen_adj = None
        if self.use_patch_pe:
            coarsen_adj = self.reshape(data.coarsen_adj)
        subgraph_x = scatter(x, batch_x, dim=0, reduce=self.pooling)
        mixer_x = self.reshape(subgraph_x)
        mixer_x = self.mlp_mixer(mixer_x, coarsen_adj)
        mixer_x = self.reshape2(mixer_x)[data.subgraphs_batch]

        # Readout
        x = torch.cat([x, mixer_x], dim=-1)
        x = scatter(x, data.subgraphs_nodes_mapper, dim=0, reduce=self.pooling)
        root_nodes = x[data.root_mask]
        x = self.output_decoder(root_nodes)
        return x


class MPGNN4TreeNeighbour(nn.Module):
    def __init__(self,
                 nfeat_node, nfeat_edge,
                 nhid, nout,
                 nlayer_gnn,
                 node_type, edge_type,
                 gnn_type,
                 rw_dim=0,
                 lap_dim=0,
                 dropout=0,
                 bn=True,
                 res=True,
                 pooling='mean'):

        super().__init__()
        self.use_rw = rw_dim > 0
        self.use_lap = lap_dim > 0
        self.pooling = pooling

        if self.use_rw:
            self.rw_encoder = MLP(rw_dim, nhid, 1)
        if self.use_lap:
            self.lap_encoder = MLP(lap_dim, nhid, 1)

        self.layer0_keys = FeatureEncoder(node_type, nfeat_node, nhid)
        self.layer0_values = FeatureEncoder(node_type, nfeat_node, nhid)
        self.edge_encoder = FeatureEncoder(edge_type, nfeat_edge, nhid)

        self.gnns = GNN(nin=nhid, nout=nhid, nlayer_gnn=nlayer_gnn,
                        gnn_type=gnn_type, bn=bn, dropout=dropout, res=res)

        self.output_decoder = MLP(
            nhid, nout, nlayer=2, with_final_activation=False)

    def forward(self, data):
        x = data.x
        x_key, x_val = x[:, 0], x[:, 1]
        x_key_embed = self.layer0_keys(x_key)
        x_val_embed = self.layer0_values(x_val)
        x = x_key_embed + x_val_embed

        if self.use_rw:
            x += self.rw_encoder(data.rw_pos_enc)
        if self.use_lap:
            x += self.lap_encoder(data.lap_pos_enc)

        edge_attr = data.edge_attr
        if edge_attr is None:
            edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))
        edge_attr = self.edge_encoder(edge_attr)

        x = self.gnns(x, data.edge_index, edge_attr)

        # node leval task
        root_nodes = x[data.root_mask]
        x = self.output_decoder(root_nodes)

        return x
