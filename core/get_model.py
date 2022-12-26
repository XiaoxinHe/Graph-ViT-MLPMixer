# from core.model import GraphMLPMixer, MPGNN
from core.model import *


def create_model(cfg):
    if cfg.dataset == 'ZINC':
        node_type = 'Discrete'
        edge_type = 'Discrete'
        nfeat_node = 28
        nfeat_edge = 4
        nout = 1  # regression

    elif cfg.dataset == 'MNIST' or cfg.dataset == 'CIFAR10':
        node_type = 'Linear'
        edge_type = 'Discrete'
        nfeat_node = 5 if cfg.dataset == 'CIFAR10' else 3
        nfeat_edge = 1
        nout = 10

    elif 'ogbg' in cfg.dataset:
        nfeat_node = None
        nfeat_edge = None
        node_type = 'Atom'
        edge_type = 'Bond'
        if cfg.dataset == 'ogbg-moltox21':
            nout = 12
        elif cfg.dataset == 'ogbg-molhiv':
            nout = 1
        elif cfg.dataset == 'ogbg-molpcba':
            nout = 128

    elif 'peptides' in cfg.dataset:
        nfeat_node = None
        nfeat_edge = None
        node_type = 'Atom'
        edge_type = 'Bond'
        nout = 10 if cfg.dataset == 'peptides-func' else 11

    elif cfg.dataset == 'CSL':
        node_type = 'Discrete'
        edge_type = 'Discrete'
        nfeat_node = 1
        nfeat_edge = 1
        nout = 10

    elif cfg.dataset == 'sr25-classify':
        nfeat_node = 2
        nfeat_edge = 1
        node_type = 'Discrete'
        edge_type = 'Discrete'
        nout = 15

    elif cfg.dataset == 'exp-classify':
        nfeat_node = 2
        nfeat_edge = 1
        node_type = 'Discrete'
        edge_type = 'Discrete'
        nout = 2

    elif cfg.dataset == 'TreeDataset':
        nfeat_node = 2 << cfg.depth
        nfeat_edge = 1
        node_type = 'Discrete'
        edge_type = 'Discrete'
        nout = 2 << cfg.depth

    if cfg.model.name == 'GraphMLPMixer':
        return GraphMLPMixer(nfeat_node=nfeat_node,
                             nfeat_edge=nfeat_edge,
                             nhid=cfg.model.hidden_size,
                             nout=nout,
                             nlayer_gnn=cfg.model.nlayer_gnn,
                             node_type=node_type,
                             edge_type=edge_type,
                             nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
                             gnn_type=cfg.model.gnn_type,
                             rw_dim=cfg.pos_enc.rw_dim,
                             lap_dim=cfg.pos_enc.lap_dim,
                             pooling=cfg.model.pool,
                             dropout=cfg.train.dropout,
                             mlpmixer_dropout=cfg.train.mlpmixer_dropout,
                             n_patches=cfg.metis.n_patches,
                             use_patch_pe=cfg.model.use_patch_pe)
    elif cfg.model.name == 'GraphViT':
        return GraphViT(nfeat_node=nfeat_node,
                        nfeat_edge=nfeat_edge,
                        nhid=cfg.model.hidden_size,
                        nout=nout,
                        nlayer_gnn=cfg.model.nlayer_gnn,
                        node_type=node_type,
                        edge_type=edge_type,
                        nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
                        gnn_type=cfg.model.gnn_type,
                        rw_dim=cfg.pos_enc.rw_dim,
                        lap_dim=cfg.pos_enc.lap_dim,
                        pooling=cfg.model.pool,
                        dropout=cfg.train.dropout,
                        mlpmixer_dropout=cfg.train.mlpmixer_dropout,
                        n_patches=cfg.metis.n_patches,
                        use_patch_pe=cfg.model.use_patch_pe)
    elif cfg.model.name == 'MPGNN':
        return MPGNN(
            nfeat_node=nfeat_node,
            nfeat_edge=nfeat_edge,
            nhid=cfg.model.hidden_size,
            nout=nout,
            nlayer_gnn=cfg.model.nlayer_gnn,
            node_type=node_type,
            edge_type=edge_type,
            gnn_type=cfg.model.gnn_type,
            rw_dim=cfg.pos_enc.rw_dim,
            lap_dim=cfg.pos_enc.lap_dim,
            pooling=cfg.model.pool,
            dropout=cfg.train.dropout)
    elif cfg.model.name == 'GraphMLPMixer4TreeNeighbour':
        return GraphMLPMixer4TreeNeighbour(
            nfeat_node=nfeat_node,
            nfeat_edge=nfeat_edge,
            nhid=cfg.model.hidden_size,
            nout=nout,
            nlayer_gnn=cfg.model.nlayer_gnn,
            node_type=node_type,
            edge_type=edge_type,
            nlayer_mlpmixer=cfg.model.nlayer_mlpmixer,
            gnn_type=cfg.model.gnn_type,
            rw_dim=cfg.pos_enc.rw_dim,
            lap_dim=cfg.pos_enc.lap_dim,
            pooling=cfg.model.pool,
            dropout=cfg.train.dropout,
            mlpmixer_dropout=cfg.train.mlpmixer_dropout,
            n_patches=cfg.metis.n_patches,
            use_patch_pe=cfg.model.use_patch_pe
        )
    elif cfg.model.name == 'MPGNN4TreeNeighbour':
        return MPGNN4TreeNeighbour(
            nfeat_node=nfeat_node,
            nfeat_edge=nfeat_edge,
            nhid=cfg.model.hidden_size,
            nout=nout,
            nlayer_gnn=cfg.model.nlayer_gnn,
            node_type=node_type,
            edge_type=edge_type,
            gnn_type=cfg.model.gnn_type,
            rw_dim=cfg.pos_enc.rw_dim,
            lap_dim=cfg.pos_enc.lap_dim,
            pooling=cfg.model.pool,
            dropout=cfg.train.dropout)
