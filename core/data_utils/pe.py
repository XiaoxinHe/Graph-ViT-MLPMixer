import torch_geometric
import torch
import numpy as np
from scipy import sparse as sp


def add_positional_encoding(data, rw_dim=0, lap_dim=0):
    if rw_dim > 0:
        data.rw_pos_enc = random_walk_positional_encoding(
            data.edge_index, rw_dim, data.num_nodes)
    if lap_dim > 0:
        data.lap_pos_enc = lap_positional_encoding(data, lap_dim)
    return data


def random_walk_positional_encoding(edge_index, pos_enc_dim, num_nodes):
    """
        Initializing positional encoding with RWPE
    """
    if edge_index.size(-1) == 0:
        PE = torch.zeros(num_nodes, pos_enc_dim)
    else:
        A = torch_geometric.utils.to_dense_adj(
            edge_index, max_num_nodes=num_nodes)[0]
        # Geometric diffusion features with Random Walk
        Dinv = A.sum(dim=-1).clamp(min=1).pow(-1).unsqueeze(-1)  # D^-1
        RW = A * Dinv
        M = RW
        M_power = M
        # Iterate
        nb_pos_enc = pos_enc_dim
        PE = [torch.diagonal(M)]
        for _ in range(nb_pos_enc-1):
            M_power = torch.matmul(M_power, M)
            PE.append(torch.diagonal(M_power))
        PE = torch.stack(PE, dim=-1)

    return PE


def lap_positional_encoding(g, pos_enc_dim):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    degree = torch_geometric.utils.degree(g.edge_index[0], g.num_nodes)
    A = torch_geometric.utils.to_scipy_sparse_matrix(
        g.edge_index, num_nodes=g.num_nodes)
    N = sp.diags(np.array(degree.clip(1) ** -0.5, dtype=float))
    L = sp.eye(g.num_nodes) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float()
    if pos_enc.size(1) < pos_enc_dim:
        zeros = torch.zeros(g.num_nodes, pos_enc_dim)
        zeros[:, :pos_enc.size(1)] = pos_enc
        pos_enc = zeros
    return pos_enc
