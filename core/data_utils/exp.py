"""
EXP dataset
"""
import torch
from torch_geometric.data import InMemoryDataset, Data
import pickle
import os


class PlanarSATPairsDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(PlanarSATPairsDataset, self).__init__(
            root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["GRAPHSAT.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(
            open(os.path.join(self.root, "raw/GRAPHSAT.pkl"), "rb"))

        data_list = [Data(**g.__dict__) for g in data_list]
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


if __name__ == "__main__":
    test_path = "dataset/EXP/"
    dataset = PlanarSATPairsDataset(test_path)
    print(dataset[0])
