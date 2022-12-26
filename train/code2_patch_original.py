import torch
import numpy as np
from core.config import cfg, update_cfg
from core.train_helper import run
import torch_geometric.transforms as T
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from functools import partial
from core.data_utils.ogbg_code2_utils import idx2vocab, get_vocab_mapping, augment_edge, encode_y_to_arr, clip_graphs_to_size, pre_transform_in_memory, decode_arr_to_seq

from train.metrics_ogb import subtoken_cross_entropy


from core.model import GraphMLPMixer_code2

from core.transform import MetisPartitionTransform


def create_dataset(cfg):
    print("[!] Loading data")
    transform = MetisPartitionTransform(cfg.metis.n_patches,
                                        recursive=cfg.metis.recursive,
                                        rw_pos_enc_dim=0,
                                        lap_pos_enc_dim=0,
                                        random=cfg.metis.online,
                                        drop_rate=cfg.metis.drop_rate,
                                        overlap=cfg.metis.overlap,
                                        transform_coarsen_adj=cfg.metis.transform_coarsen_adj if cfg.model.weighted_mlpmixer else False,
                                        min_nodes_per_patch=cfg.metis.min_nodes_per_patch,
                                        num_hops=cfg.metis.num_hops)

    transform_eval = MetisPartitionTransform(cfg.metis.n_patches,
                                             recursive=cfg.metis.recursive,
                                             rw_pos_enc_dim=0,
                                             lap_pos_enc_dim=0,
                                             random=False,
                                             drop_rate=0.0,
                                             overlap=cfg.metis.overlap,
                                             transform_coarsen_adj=cfg.metis.transform_coarsen_adj if cfg.model.weighted_mlpmixer else False,
                                             min_nodes_per_patch=cfg.metis.min_nodes_per_patch,
                                             num_hops=cfg.metis.num_hops)

    name = 'ogbg-code2'
    dataset_dir = 'mydata'
    dataset = PygGraphPropPredDataset(name=name, root=dataset_dir)
    s_dict = dataset.get_idx_split()
    dataset.split_idxs = [s_dict[s] for s in ['train', 'valid', 'test']]

    num_vocab = 5000  # The number of vocabulary used for sequence prediction
    max_seq_len = 5  # The maximum sequence length to predict
    seq_len_list = np.array([len(seq) for seq in dataset.data.y])
    print(f"Target sequences less or equal to {max_seq_len} is "
          f"{np.sum(seq_len_list <= max_seq_len) / len(seq_len_list)}")

    # Building vocabulary for sequence prediction. Only use training data.
    vocab2idx, idx2vocab_local = get_vocab_mapping(
        [dataset.data.y[i] for i in s_dict['train']], num_vocab)
    print(f"Final size of vocabulary is {len(vocab2idx)}")
    # Set to global variable to later access in CustomLogger
    idx2vocab.extend(idx2vocab_local)

    # Set the transform function:
    # augment_edge: add next-token edge as well as inverse edges. add edge attributes.
    # encode_y_to_arr: add y_arr to PyG data object, indicating the array repres
    transform = T.Compose(
        [augment_edge,
         lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len),
         # Subset graphs to a maximum size (number of nodes) limit.
         #  partial(clip_graphs_to_size, size_limit=1000),
         transform])
    transform_eval = T.Compose(
        [augment_edge,
         lambda data: encode_y_to_arr(data, vocab2idx, max_seq_len),
         # Subset graphs to a maximum size (number of nodes) limit.
         #  partial(clip_graphs_to_size, size_limit=1000),
         transform_eval])

    # Subset graphs to a maximum size (number of nodes) limit.
    pre_transform_in_memory(dataset, partial(
        clip_graphs_to_size, size_limit=1000))

    split_idx = dataset.get_idx_split()
    train_dataset, val_dataset, test_dataset = dataset[split_idx['train']
                                                       ], dataset[split_idx['valid']], dataset[split_idx['test']]
    train_dataset.transform, val_dataset.transform, test_dataset.transform = transform, transform_eval, transform_eval

    return train_dataset, val_dataset, test_dataset


def create_model(cfg):
    print("[!] Loading model")
    model = GraphMLPMixer_code2(nfeat_node=None,
                                nfeat_edge=None,
                                nhid=cfg.model.hidden_size,
                                nout=5002,  # len(vocab2idx)
                                nlayer_gnn=cfg.model.num_layers,
                                nlayer_mlpmixer=cfg.model.num_mlpmixer_layers,
                                gnn_type=cfg.model.gnn_type,
                                rw_pos_enc_dim=cfg.pos_enc.rw_dim,
                                lap_pos_enc_dim=cfg.pos_enc.lap_dim,
                                pooling=cfg.model.pool,
                                dropout=cfg.train.dropout,
                                n_patches=cfg.metis.n_patches,
                                weighted_mlpmixer=cfg.model.weighted_mlpmixer)
    return model


def train(train_loader, model, optimizer, device, sharp=False):
    model.train()
    seq_ref_list = []
    seq_pred_list = []

    for i, data in enumerate(train_loader):
        if model.lap_pos_enc:
            batch_pos_enc = data.lap_pos_enc
            sign_flip = torch.rand(batch_pos_enc.size(1))
            sign_flip[sign_flip >= 0.5] = 1.0
            sign_flip[sign_flip < 0.5] = -1.0
            data.lap_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
        data = data.to(device)
        optimizer.zero_grad()
        pred, true = model(data)
        loss, pred_list = subtoken_cross_entropy(pred, true)
        loss.backward()
        optimizer.step()

        assert true['y_arr'].shape[1] == len(pred)  # max_seq_len (5)
        assert true['y_arr'].shape[0] == pred[0].shape[0]  # batch size

        # Decode the predicted sequence tokens, so we don't need to store
        # the logits that take significant memory.

        def arr_to_seq(arr): return decode_arr_to_seq(arr, idx2vocab)
        mat = []
        for i in range(len(pred_list)):
            mat.append(torch.argmax(pred_list[i].detach(), dim=1).view(-1, 1))
        mat = torch.cat(mat, dim=1)

        seq_pred = [arr_to_seq(arr) for arr in mat]
        seq_ref = [true['y'][i] for i in range(len(true['y']))]
        seq_ref_list.extend(seq_ref)
        seq_pred_list.extend(seq_pred)

    score = evaluator.eval(
        {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list})['F1']
    return score, loss


def test(loader, model, evaluator, device):
    model.eval()
    seq_ref_list = []
    seq_pred_list = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            pred, true = model(data)
            loss, pred_list = subtoken_cross_entropy(pred, true)
            assert true['y_arr'].shape[1] == len(pred)  # max_seq_len (5)
            assert true['y_arr'].shape[0] == pred[0].shape[0]  # batch size

            # Decode the predicted sequence tokens, so we don't need to store
            # the logits that take significant memory.

            def arr_to_seq(arr): return decode_arr_to_seq(arr, idx2vocab)
            mat = []
            for i in range(len(pred_list)):
                mat.append(torch.argmax(
                    pred_list[i].detach(), dim=1).view(-1, 1))
            mat = torch.cat(mat, dim=1)

            seq_pred = [arr_to_seq(arr) for arr in mat]
            seq_ref = [true['y'][i] for i in range(len(true['y']))]
            seq_ref_list.extend(seq_ref)
            seq_pred_list.extend(seq_pred)
    score = evaluator.eval(
        {"seq_ref": seq_ref_list, "seq_pred": seq_pred_list})['F1']
    return score, loss


if __name__ == '__main__':
    # get config
    cfg.merge_from_file('train/configs/code2_patch.yaml')
    cfg = update_cfg(cfg)

    evaluator = Evaluator(name=cfg.dataset)
    run(cfg, create_dataset, create_model, train, test, evaluator=evaluator)
