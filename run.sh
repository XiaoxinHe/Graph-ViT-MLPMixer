#!/bin/bash


################################################################ CSL ################################################################
# MPGNN
CONFIG='train/configs/MPGNN/csl.yaml'
python -m train.csl --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.csl --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.csl --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.csl --config $CONFIG device 3 model.gnn_type TransformerConv   ;

# Graph MLPMixer
CONFIG='train/configs/GraphMLPMixer/csl.yaml'
python -m train.csl --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.csl --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.csl --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.csl --config $CONFIG device 3 model.gnn_type TransformerConv   ;
#######################################################################################################################################


################################################################ EXP ################################################################
# MPGNN
CONFIG='train/configs/MPGNN/exp.yaml'
python -m train.exp --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.exp --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.exp --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.exp --config $CONFIG device 3 model.gnn_type TransformerConv   ;

# Graph MLPMixer
CONFIG='train/configs/GraphMLPMixer/exp.yaml'
python -m train.exp --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.exp --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.exp --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.exp --config $CONFIG device 3 model.gnn_type TransformerConv   ;
#######################################################################################################################################


################################################################ SR25 ################################################################
# MPGNN
CONFIG='train/configs/MPGNN/sr25.yaml'
python -m train.sr25 --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.sr25 --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.sr25 --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.sr25 --config $CONFIG device 3 model.gnn_type TransformerConv   ;

# Graph MLPMixer
CONFIG='train/configs/GraphMLPMixer/sr25.yaml'
python -m train.sr25 --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.sr25 --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.sr25 --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.sr25 --config $CONFIG device 3 model.gnn_type TransformerConv   ;
#######################################################################################################################################


################################################################ ZINC ################################################################
# MPGNN
CONFIG='train/configs/MPGNN/zinc.yaml'
python -m train.zinc --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.zinc --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.zinc --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.zinc --config $CONFIG device 3 model.gnn_type TransformerConv   ;

# Graph MLPMixer
CONFIG='train/configs/GraphMLPMixer/zinc.yaml'
python -m train.zinc --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.zinc --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.zinc --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.zinc --config $CONFIG device 3 model.gnn_type TransformerConv   ;
#######################################################################################################################################


################################################################ MNIST ################################################################
# MPGNN
CONFIG='train/configs/MPGNN/mnist.yaml'
python -m train.mnist --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.mnist --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.mnist --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.mnist --config $CONFIG device 3 model.gnn_type TransformerConv   ;

# Graph MLPMixer
CONFIG='train/configs/GraphMLPMixer/mnist.yaml'
python -m train.mnist --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.mnist --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.mnist --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.mnist --config $CONFIG device 3 model.gnn_type TransformerConv   ;
#######################################################################################################################################


################################################################ CIFAR10 ################################################################
# MPGNN
CONFIG='train/configs/MPGNN/cifar10.yaml'
python -m train.cifar10 --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.cifar10 --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.cifar10 --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.cifar10 --config $CONFIG device 3 model.gnn_type TransformerConv   ;

# Graph MLPMixer
CONFIG='train/configs/GraphMLPMixer/cifar10.yaml'
python -m train.cifar10 --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.cifar10 --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.cifar10 --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.cifar10 --config $CONFIG device 3 model.gnn_type TransformerConv   ;
#######################################################################################################################################


################################################################ MolTox21 ################################################################
# MPGNN
CONFIG='train/configs/MPGNN/moltox21.yaml'
python -m train.moltox21 --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.moltox21 --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.moltox21 --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.moltox21 --config $CONFIG device 3 model.gnn_type TransformerConv   ;

# Graph MLPMixer
CONFIG='train/configs/GraphMLPMixer/moltox21.yaml'
python -m train.moltox21 --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.moltox21 --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.moltox21 --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.moltox21 --config $CONFIG device 3 model.gnn_type TransformerConv   ;
#######################################################################################################################################

################################################################ MolHIV ################################################################
# MPGNN
CONFIG='train/configs/MPGNN/molhiv.yaml'
python -m train.molhiv --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.molhiv --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.molhiv --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.molhiv --config $CONFIG device 3 model.gnn_type TransformerConv   ;

# Graph MLPMixer
CONFIG='train/configs/GraphMLPMixer/molhiv.yaml'
python -m train.molhiv --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.molhiv --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.molhiv --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.molhiv --config $CONFIG device 3 model.gnn_type TransformerConv   ;
#######################################################################################################################################


################################################################ Peptides-func ################################################################
# MPGNN
CONFIG='train/configs/MPGNN/peptides_func.yaml'
python -m train.peptides_func --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.peptides_func --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.peptides_func --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.peptides_func --config $CONFIG device 3 model.gnn_type TransformerConv   ;

# Graph MLPMixer
CONFIG='train/configs/GraphMLPMixer/peptides_func.yaml'
python -m train.peptides_func --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.peptides_func --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.peptides_func --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.peptides_func --config $CONFIG device 3 model.gnn_type TransformerConv   ;
#######################################################################################################################################


################################################################ Peptides-struct ################################################################
# MPGNN
CONFIG='train/configs/MPGNN/peptides_struct.yaml'
python -m train.peptides_struct --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.peptides_struct --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.peptides_struct --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.peptides_struct --config $CONFIG device 3 model.gnn_type TransformerConv   ;

# Graph MLPMixer
CONFIG='train/configs/GraphMLPMixer/peptides_struct.yaml'
python -m train.peptides_struct --config $CONFIG device 0 model.gnn_type ResGatedGraphConv &
python -m train.peptides_struct --config $CONFIG device 1 model.gnn_type GINEConv          &
python -m train.peptides_struct --config $CONFIG device 2 model.gnn_type GCNConv           &
python -m train.peptides_struct --config $CONFIG device 3 model.gnn_type TransformerConv   ;
#######################################################################################################################################


################################################################ TreeNeighbourMatch ################################################################
# MPGNN
CONFIG='train/configs/MPGNN/tree_neighbour.yaml'
for DEPTH in 2 3 4 5 6 7 8
do 
    NLAYER_GNN=$(($DEPTH+1))
    python -m train.tree_neighbour --config $CONFIG depth $DEPTH model.nlayer_gnn $NLAYER_GNN device 0 model.gnn_type GATConv           &
    python -m train.tree_neighbour --config $CONFIG depth $DEPTH model.nlayer_gnn $NLAYER_GNN device 1 model.gnn_type GINEConv          &
    python -m train.tree_neighbour --config $CONFIG depth $DEPTH model.nlayer_gnn $NLAYER_GNN device 2 model.gnn_type GCNConv           &
    python -m train.tree_neighbour --config $CONFIG depth $DEPTH model.nlayer_gnn $NLAYER_GNN device 3 model.gnn_type GatedGraphConv    ;
done

# Graph MLPMixer
CONFIG='train/configs/GraphMLPMixer/tree_neighbour.yaml'
DEPTH=2
python -m train.tree_neighbour --config $CONFIG depth $DEPTH metis.n_patches 8 train.dropout 0.5 device 0 model.gnn_type GATConv           &
python -m train.tree_neighbour --config $CONFIG depth $DEPTH metis.n_patches 8 train.dropout 0.5 device 1 model.gnn_type GINEConv          &
python -m train.tree_neighbour --config $CONFIG depth $DEPTH metis.n_patches 8 train.dropout 0.5 device 2 model.gnn_type GCNConv           &
python -m train.tree_neighbour --config $CONFIG depth $DEPTH metis.n_patches 8 train.dropout 0.5 device 3 model.gnn_type GatedGraphConv    ;
for DEPTH in 3 4 5 6 7 8
do
    NLAYER_GNN=$(($DEPTH+1))
    python -m train.tree_neighbour --config $CONFIG depth $DEPTH model.nlayer_gnn $NLAYER_GNN device 0 model.gnn_type GATConv           &
    python -m train.tree_neighbour --config $CONFIG depth $DEPTH model.nlayer_gnn $NLAYER_GNN device 1 model.gnn_type GINEConv          &
    python -m train.tree_neighbour --config $CONFIG depth $DEPTH model.nlayer_gnn $NLAYER_GNN device 2 model.gnn_type GCNConv           &
    python -m train.tree_neighbour --config $CONFIG depth $DEPTH model.nlayer_gnn $NLAYER_GNN device 3 model.gnn_type GatedGraphConv    ;
done
####################################################################################################################################################