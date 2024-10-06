from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import torch_geometric as tg
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset

class AssetDataset():
    def __init__(self, args, features, adjs, shortest_paths, edge_feat):
        super(AssetDataset, self).__init__()
        self.args = args
        self.max_steps = len(adjs)
        self.labels = [self._preprocess_labels(feat) for feat in features] # (t, n), (time, #nodes)
        self.features = self._preprocess_features(features) # (t, n, dim)
        self.adjs = [self._preprocess_adj(a) for a in adjs] # (t,n,n)
        self.edge_feat = edge_feat
        self.shortest_paths = shortest_paths
        self.hist_time_steps = args.hist_time_steps
        self._split_data()

    def _preprocess_labels(self, features):
        """split return (label) from features"""
        features = np.array(features.todense())
        labels = features[:,0]
        return labels

    def _preprocess_features(self, features):
        """min max normalization on features"""
        features = np.array([np.array(i.todense()) for i in features])
        if self.args.feat_norm:
            for i in range(1,features.shape[-1]):
                maxi = features[:,:,i].max()
                mini = features[:,:,i].min()
                if maxi-mini != 0:
                    features[:,:,i] = (features[:,:,i] - mini)/(maxi-mini)
            return features
        return features

    def _preprocess_adj(self, adj):
        if self.args.adj_norm:
            """normalization of adjacency matrix"""
            rowsum = np.array(adj.sum(1))
            r_inv = sp.diags(np.power(rowsum, -1).flatten(), dtype=np.float32)
            adj_normalized = r_inv.dot(adj)
            return adj_normalized
        return adj

    def _split_data(self):
        train_start = self.args.hist_time_steps
        valid_start = int(np.floor(self.max_steps * self.args.train_proportion)) 
        test_start = int(np.floor(self.max_steps * (self.args.train_proportion + self.args.valid_proportion)))
        train = AssetBatch(self.args, self.adjs, self.features, self.labels, self.shortest_paths, self.edge_feat,
                           train_start, valid_start-1) 
        valid = AssetBatch(self.args, self.adjs, self.features, self.labels, self.shortest_paths, self.edge_feat,
                           valid_start + self.args.hist_time_steps, test_start-1) 
        test = AssetBatch(self.args, self.adjs, self.features, self.labels, self.shortest_paths, self.edge_feat,
                           test_start + self.args.hist_time_steps, self.max_steps-1) 
        
        self.train = DataLoader(train, shuffle=True)
        self.valid = DataLoader(valid, shuffle=False)
        self.test = DataLoader(test, shuffle=False)

        print('Dataset splits: ')
        print('{:<3} train samples from {:<3} to {:<3}'.format(len(self.train), train_start, valid_start-1))
        print('{:<3} valid samples from {:<3} to {:<3}'.format(len(self.valid), valid_start + self.args.hist_time_steps, test_start-1))
        print('{:<3} test samples from {:<3} to {:<3}'.format(len(self.test), test_start+ self.args.hist_time_steps, self.max_steps-1))
            

class AssetBatch(Dataset):
    def  __init__(self, args, adjs, features, labels,  shortest_paths, edge_feat,
                  start, end):
        super(AssetBatch, self).__init__()
        self.args = args
        self.start = start
        self.end = end
        self.adjs = adjs
        self.features = features
        self.labels = labels
        self.edge_feat = edge_feat
        self.shortest_paths = shortest_paths
        self.hist_time_steps = args.hist_time_steps

    def __len__(self):
        return self.end-self.start + 1

    def __getitem__(self, index):
        idx = index+self.start 
        hist_graphs = []
        for i in range(idx - self.args.hist_time_steps, idx):
            x = torch.Tensor(self.features[i]) # N, F 
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(self.adjs[i])
            shortest_path_len = self.shortest_paths[i]
            edge_feat = torch.Tensor(self.edge_feat[i])
            graph = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, edge_feat = edge_feat,
                         shortest_path_len = shortest_path_len)
            hist_graphs.append(graph)
        sample = {'idx': torch.Tensor([idx]), 
                'hist_graphs': hist_graphs, # [g0,...,g11]
                'labels': torch.Tensor(self.labels[idx])}
        return sample
