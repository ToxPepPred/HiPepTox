import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch_scatter import scatter_mean
import torch_geometric
from torch_geometric.nn import radius_graph, TransformerConv


class ProteinGraphDataset(data.Dataset):
    def __init__(self, seq_list, label_list, outpath, radius=15):
        super(ProteinGraphDataset, self).__init__()
        self.seq_list = seq_list
        self.label_list = label_list
        self.path = outpath
        self.radius = radius

    def __len__(self): return len(self.seq_list)

    def __getitem__(self, idx): return self._featurize_graph(idx)

    def _featurize_graph(self, idx):
        sequence = self.seq_list[idx]
        with torch.no_grad():
            X = torch.load(self.path + "structure_local/" + sequence + ".tensor", map_location=torch.device('cpu'))
            # note: the esm_feat is on the device which is used to calculate the feature
            esm_feat = torch.load(self.path + "esm_embedding/" + sequence + ".pt", map_location=torch.device('cpu'))
            pre_computed_node_feat = esm_feat

            X_ca = X[:, 1]
            edge_index = radius_graph(X_ca, r=self.radius, loop=True, max_num_neighbors=1000, num_workers=8)

        graph_data = torch_geometric.data.Data(name=sequence, X=X, y=torch.tensor(self.label_list[idx]),
                                               node_feat=pre_computed_node_feat, edge_index=edge_index)
        return graph_data
