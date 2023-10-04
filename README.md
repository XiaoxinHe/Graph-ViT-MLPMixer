# Graph-ViT-MLPMixer

[![arXiv](https://img.shields.io/badge/arXiv-2212.13350-b31b1b.svg)](https://arxiv.org/abs/2212.13350) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-generalization-of-vit-mlp-mixer-to-graphs/graph-classification-on-peptides-func)](https://paperswithcode.com/sota/graph-classification-on-peptides-func?p=a-generalization-of-vit-mlp-mixer-to-graphs) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-generalization-of-vit-mlp-mixer-to-graphs/graph-regression-on-peptides-struct)](https://paperswithcode.com/sota/graph-regression-on-peptides-struct?p=a-generalization-of-vit-mlp-mixer-to-graphs) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-generalization-of-vit-mlp-mixer-to-graphs/graph-regression-on-zinc)](https://paperswithcode.com/sota/graph-regression-on-zinc?p=a-generalization-of-vit-mlp-mixer-to-graphs)

# Citation

```
@inproceedings{he2023generalization,
  title={A generalization of vit/mlp-mixer to graphs},
  author={He, Xiaoxin and Hooi, Bryan and Laurent, Thomas and Perold, Adam and LeCun, Yann and Bresson, Xavier},
  booktitle={International Conference on Machine Learning},
  pages={12724--12745},
  year={2023},
  organization={PMLR}
}
```

# Python environment setup with Conda

```
conda create --name graph_mlpmixer python=3.8
conda activate graph_mlpmixer

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install -c pyg pytorch-sparse
conda install -c pyg pytorch-scatter
conda install -c pyg pytorch-cluster
conda install -c pyg pyg
pip install ogb
conda install -c conda-forge rdkit
pip install yacs
pip install tensorboard
pip install networkx
pip install einops

# METIS
conda install -c conda-forge metis
pip install metis
```

# Run Graph MLPMixer

## Run different datasets

See all available datasets under the `train/` folder.

```
# Running Graph MLPMixer on simulation datasets
python -m train.exp

# Running Graph MLPMixer for ZINC
python -m train.zinc

# Running Graph MLPMixer for CIFAR10
python -m train.cifar10

# Running Graph MLPMixer on OGB datasets
python -m train.molhiv

# Running Graph MLPMixer on LRGB datasets
python -m train.peptides_func
```

## Run different base MP-GNNs

See all available base MP-GNNs in `core/model_utils/pyg_gnn_wrapper.py`.

```
python -m train.zinc model.gnn_type GCNConv
python -m train.zinc model.gnn_type ResGatedGraphConv
...
```





## Run ablation studies

See `core/config.py` for all options.

```
# Running Graph MLPMixer w/o NodePE
python -m train.zinc pos_enc.rw_dim 0 pos_enc.lap_dim 0

# Running Graph MLPMixer w/o PatchPE
python -m train.zinc model.patch_rw_dim 0

# Running Graph MLPMixer w/o data augmentation
python -m train.zinc metis.online False

# Running Graph MLPMixer w/o patch overlapping
python -m train.zinc metis.num_hops 0

# Running Graph MLPMixer w/ k-hop extension (replace k with 1,2,...)
python -m train.zinc metis.num_hops k

# Running Graph MLPMixer using random partition
python -m train.zinc metis.enable False

# Running Graph MLPMixer w/ different numbers of Patches P
python -m train.zinc metis.n_patches P
```

# Run Graph ViT
We provide five different versions of Graph ViT. The versions primarily differ in the graph-based multi-head attention (gMHA) function used. 
```
# Hadamard Attention
python -m train.zinc model.gMHA_type Hadamard pos_enc.patch_num_diff 0

# Standard/Full Attention
python -m train.zinc model.gMHA_type Standard pos_enc.patch_num_diff -1

# Graph Attention
python -m train.zinc model.gMHA_type Graph pos_enc.patch_num_diff 0

# Kernel Attention
python -m train.zinc model.gMHA_type Kernel pos_enc.patch_num_diff 8

# Additive Attention 
python -m train.zinc model.gMHA_type Additive pos_enc.patch_num_diff 0
```

# Run MP-GNNs

Run MP-GNNs by setting n_patches to zero.

```
python -m train.zinc model.name MPGNN metis.n_patches 0
python -m train.peptides_func model.name MPGNN metis.n_patches 0
...
```

# Reproducibility

Use `run.sh` to run the codes and reproduce the published results.
