import os
import argparse
from yacs.config import CfgNode as CN


def set_cfg(cfg):

    # ------------------------------------------------------------------------ #
    # Basic options
    # ------------------------------------------------------------------------ #
    # Dataset name
    cfg.dataset = 'ZINC'
    # Additional num of worker for data loading
    cfg.num_workers = 8
    # Cuda device number, used for machine with multiple gpus
    cfg.device = 0
    # Whether fix the running seed to remove randomness
    cfg.seed = None
    # Custom log file name
    cfg.logfile = None
    # tree depth for TreeDataset
    cfg.depth = -1

    # ------------------------------------------------------------------------ #
    # Training options
    # ------------------------------------------------------------------------ #
    cfg.train = CN()
    # Total graph mini-batch size
    cfg.train.batch_size = 128
    # Maximal number of epochs
    cfg.train.epochs = 1000
    # Number of runs with random init
    cfg.train.runs = 4
    # Base learning rate
    cfg.train.lr = 0.001
    # number of steps before reduce learning rate
    cfg.train.lr_patience = 20
    # learning rate decay factor
    cfg.train.lr_decay = 0.5
    # L2 regularization, weight decay
    cfg.train.wd = 0.
    # Dropout rate
    cfg.train.dropout = 0.
    # Dropout rate for MLPMixer
    cfg.train.mlpmixer_dropout = 0.
    # A lower bound on the learning rate.
    cfg.train.min_lr = 1e-5
    # optimizer
    cfg.train.optimizer = 'Adam'

    # ------------------------------------------------------------------------ #
    # Model options
    # ------------------------------------------------------------------------ #
    cfg.model = CN()
    # GNN type used, see core.model_utils.pyg_gnn_wrapper for all options
    cfg.model.gnn_type = 'GINEConv'  # change to list later
    # GraphMLPMixer or graph-based multihead attention: [MLPMixer, Hadamard, Standard, Graph, Addictive, Kernel]
    cfg.model.gMHA_type = 'MLPMixer'
    # Hidden size of the model
    cfg.model.hidden_size = 128
    # Number of gnn layers
    cfg.model.nlayer_gnn = 4
    # Number of mlp mixer layers
    cfg.model.nlayer_mlpmixer = 4
    # Pooling type for generaating graph/subgraph embedding from node embeddings
    cfg.model.pool = 'mean'
    # Use residual connection
    cfg.model.residual = True

    # ------------------------------------------------------------------------ #
    # Positional encoding options
    # ------------------------------------------------------------------------ #
    cfg.pos_enc = CN()
    # Random walk structural encoding
    cfg.pos_enc.rw_dim = 0
    # Laplacian eigenvectors positional encoding
    cfg.pos_enc.lap_dim = 0
    # Patch random walk structural encoding
    cfg.pos_enc.patch_rw_dim = 8
    # Patch PE diffusion steps
    cfg.pos_enc.patch_num_diff = -1

    # ------------------------------------------------------------------------ #
    # Metis patch extraction options
    # ------------------------------------------------------------------------ #
    cfg.metis = CN()
    # Enable Metis partition (otherwise use random partition)
    cfg.metis.enable = True
    # Enable data augmentation
    cfg.metis.online = True
    # The number of partitions
    cfg.metis.n_patches = 32
    # Whether to randomly drop a set of edges before each metis partition
    cfg.metis.drop_rate = 0.3
    # expanding patches with k hop neighbourhood
    cfg.metis.num_hops = 1

    return cfg


# Principle means that if an option is defined in a YACS config object,
# then your program should set that configuration option using cfg.merge_from_list(opts) and not by defining,
# for example, --train-scales as a command line argument that is then used to set cfg.TRAIN.SCALES.


def update_cfg(cfg, args_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default="",
                        metavar="FILE", help="Path to config file")
    # opts arg needs to match set_cfg
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")

    if isinstance(args_str, str):
        # parse from a string
        args = parser.parse_args(args_str.split())
    else:
        # parse from command line
        args = parser.parse_args()
    # Clone the original cfg
    cfg = cfg.clone()

    # Update from config file
    if os.path.isfile(args.config):
        cfg.merge_from_file(args.config)

    # Update from command line
    cfg.merge_from_list(args.opts)

    return cfg


"""
    Global variable
"""
cfg = set_cfg(CN())
