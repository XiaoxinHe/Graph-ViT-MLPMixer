# Graph-MLPMixer



# Python environment setup with Conda


```
conda create --name graph_mlpmixer python=3.8
conda activate graph_mlpmixer
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 

python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.version.cuda)
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.11.0+cu113.html

pip install torch-geometric
pip install ogb
pip install yacs
pip install tensorboard
pip install networkx
pip install einops

# METIS
conda install -c conda-forge metis
pip install metis

# LRGB
conda install openbabel fsspec rdkit -c conda-forge
```

# Running Graph MLPMixer
## Run different datasets
See all available datasets under `train` folder.

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

## Run normal GNNs
Run normal GNNs by specifying model name and setting n_patches to zero.
```
python -m train.zinc model.name MPGNN metis.n_patches 0
python -m train.peptides_func model.name MPGNN metis.n_patches 0
...
```

## Run ablation studies
See `core/config.py` for all options.
```
# Running Graph MLPMixer wo/ NodePE
python -m train.zinc pos_enc.rw_dim 0 pos_enc.lap_dim 0

# Running Graph MLPMixer wo/ PatchPE
python -m train.zinc model.use_patch_pe False

# Running Graph MLPMixer wo/ data augmentation
python -m train.zinc metis.online False

# Running Graph MLPMixer wo/ patch overlapping
python -m train.zinc metis.num_hops 0

# Running Graph MLPMixer w/ k-hop extension (replace k with 1,2,...)
python -m train.zinc metis.num_hops k

# Running Graph MLPMixer w/ random node partition
python -m train.zinc metis.enable False

# Running Graph MLPMixer w/ different number of Patches P
python -m train.zinc metis.n_patches P
```

# Reproducibility
Use `run.sh` to run the codes and reproduce the published results.


# Citation


