import torch
import torch_geometric
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import numpy as np
import itertools
import random
import math
from sklearn.model_selection import train_test_split

"""
Adapted from https://github.com/tech-srl/bottleneck/blob/main/tasks/tree_dataset.py
"""


class TreeDataset(InMemoryDataset):
    def __init__(self, root='dataset', depth=3, add_self_loops=False, train_fraction=0.8, transform=None, pre_transform=None, pre_filter=None):
        self.depth = depth
        self.criterion = torch.nn.functional.cross_entropy
        self.num_nodes, self.edges, self.leaf_indices = self._create_blank_tree()
        self.train_fraction = train_fraction
        self.add_self_loops = add_self_loops
        super().__init__(root, transform, pre_transform, pre_filter)
        self.train = torch.load(self.processed_paths[0])
        self.val = torch.load(self.processed_paths[1])
        self.test = torch.load(self.processed_paths[2])

    def get_combinations(self):
        # returns: an iterable of [key, permutation(leaves)]
        # number of combinations: (num_leaves!)*num_choices
        num_leaves = len(self.leaf_indices)
        num_permutations = 4000
        max_examples = 640000

        if self.depth > 3:
            per_depth_num_permutations = min(num_permutations, math.factorial(
                num_leaves), max_examples // num_leaves)
            print("per_depth_num_permutations: ", per_depth_num_permutations)
            permutations = [np.random.permutation(range(1, num_leaves + 1)) for _ in
                            range(per_depth_num_permutations)]
        else:
            permutations = random.sample(list(itertools.permutations(range(1, num_leaves + 1))),
                                         min(num_permutations, math.factorial(num_leaves)))

        return itertools.chain.from_iterable(

            zip(range(1, num_leaves + 1), itertools.repeat(perm))
            for perm in permutations)

    def add_child_edges(self, cur_node, max_node):
        edges = []
        leaf_indices = []
        stack = [(cur_node, max_node)]
        while len(stack) > 0:
            cur_node, max_node = stack.pop()
            if cur_node == max_node:
                leaf_indices.append(cur_node)
                continue
            left_child = cur_node + 1
            right_child = cur_node + 1 + ((max_node - cur_node) // 2)
            edges.append([left_child, cur_node])
            edges.append([right_child, cur_node])
            stack.append((right_child, max_node))
            stack.append((left_child, right_child - 1))
        return edges, leaf_indices

    def _create_blank_tree(self):
        max_node_id = 2 ** (self.depth + 1) - 2
        edges, leaf_indices = self.add_child_edges(
            cur_node=0, max_node=max_node_id)
        return max_node_id + 1, edges, leaf_indices

    def create_blank_tree(self, add_self_loops=True):
        edge_index = torch.tensor(self.edges).t()
        if add_self_loops:
            edge_index, _ = torch_geometric.utils.add_remaining_self_loops(
                edge_index=edge_index, )
        return edge_index

    def generate_data(self):
        data_list = []
        for comb in self.get_combinations():
            edge_index = self.create_blank_tree(
                add_self_loops=self.add_self_loops)
            nodes = torch.tensor(
                self.get_nodes_features(comb), dtype=torch.long)
            root_mask = torch.tensor([True] + [False] * (len(nodes) - 1))
            label = self.label(comb)
            data = Data(x=nodes, edge_index=edge_index,
                        root_mask=root_mask, y=label)
            data_list.append(data)
        return data_list

    def get_nodes_features(self, combination):
        # combination: a list of indices
        # Each leaf contains a one-hot encoding of a key, and a one-hot encoding of the value
        # Every other node is empty, for now
        selected_key, values = combination

        # The root is [one-hot selected key] + [0 ... 0]
        nodes = [(selected_key, 0)]

        for i in range(1, self.num_nodes):
            if i in self.leaf_indices:
                leaf_num = self.leaf_indices.index(i)
                node = (leaf_num+1, values[leaf_num])
            else:
                node = (0, 0)
            nodes.append(node)
        return nodes

    def label(self, combination):
        selected_key, values = combination
        return int(values[selected_key - 1])

    def get_dims(self):
        # get input and output dims
        in_dim = len(self.leaf_indices)
        out_dim = len(self.leaf_indices)
        return in_dim, out_dim

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2']

    @property
    def processed_file_names(self):
        train = 'train_' + 'depth' + str(self.depth) + '.pt'
        val = 'val_' + 'depth' + str(self.depth) + '.pt'
        test = 'test_' + 'depth' + str(self.depth) + '.pt'
        return [train, val, test]

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        print("[!] Generating data")
        data_list = self.generate_data()

        if self.pre_filter is not None:
            data_list = [
                data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        X_train, X_test = train_test_split(
            data_list, train_size=self.train_fraction, shuffle=True, stratify=[data.y for data in data_list])
        X_train, X_val = train_test_split(
            X_train, train_size=0.75, shuffle=True, stratify=[data.y for data in X_train])
        torch.save(X_train, self.processed_paths[0])
        torch.save(X_val, self.processed_paths[1])
        torch.save(X_test, self.processed_paths[2])
