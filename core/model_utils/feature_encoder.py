import torch.nn as nn
from ogb.graphproppred.mol_encoder import AtomEncoder as AtomEncoder_
from ogb.graphproppred.mol_encoder import BondEncoder as BondEncoder_


def AtomEncoder(nin, nhid):
    return AtomEncoder_(nhid)


def BondEncoder(nin, nhid):
    return BondEncoder_(nhid)


def DiscreteEncoder(nin, nhid):
    return nn.Embedding(nin, nhid)


def LinearEncoder(nin, nhid):
    return nn.Linear(nin, nhid)


def FeatureEncoder(TYPE, nin, nhid):
    models = {
        'Atom': AtomEncoder,
        'Bond': BondEncoder,
        'Discrete': DiscreteEncoder,
        'Linear': LinearEncoder
    }

    return models[TYPE](nin, nhid)
