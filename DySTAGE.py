import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss

from models.layers import TemporalLayer, TopologicalLayer

class DySTAGE(nn.Module):
    def __init__(self, args, num_nodes, num_features, edge_scale):
        super(DySTAGE, self).__init__()
        self.args = args
        self.num_time_steps = args.hist_time_steps
        self.num_features = num_features
        self.num_nodes = num_nodes
        self.edge_scale = edge_scale

        self.spatial = args.spatial
        self.centrality = args.centrality
        self.edge = args.edge

        self.structural_n_heads = args.n_heads
        self.structural_node_dim = args.node_dim
        self.structural_n_layers = args.attention_layers
        self.temporal_head_config = list(map(int, args.temporal_head_config.split(",")))
        self.temporal_layer_config = list(map(int, args.temporal_layer_config.split(",")))
        self.spatial_drop = args.spatial_drop
        self.temporal_drop = args.temporal_drop

        self.topological, self.temporal, self.final = self.build_model()
        self.mseloss = MSELoss()

        self.chosen = 'all' if args.dataset in ['sp','ml_all','asset_all'] else None


    def forward(self, graphs):
        structural_out = torch.cat(
            [self.topological(graphs[t]).x.unsqueeze(1) for t in range(0, self.num_time_steps)],
            dim=1) # [N, T, F]
        temporal_out = self.temporal(structural_out) # [N,T,F]
        out = torch.squeeze(self.final(temporal_out[:,-1,:])) # [N]
        return out

    def build_model(self):
        input_dim = self.num_features

        # 1: Topological module
        topological_layer = TopologicalLayer(num_nodes = self.num_nodes,
                                             input_dim=input_dim,
                                             node_dim = self.structural_node_dim,
                                             edge_scale = self.edge_scale,
                                             out_dim=self.temporal_layer_config[0],
                                             n_heads=self.structural_n_heads,
                                             num_layers=self.structural_n_layers,
                                             centrality = self.centrality,
                                             spatial = self.spatial,
                                             edge = self.edge)
            
        # 2: Temporal module
        input_dim = self.temporal_layer_config[0]
        temporal_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalLayer(input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args.residual)
            temporal_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]
        
        final_layer = nn.Sequential(nn.Linear(input_dim, 1, bias=False),
                                    nn.Tanh())

        return topological_layer, temporal_layers, final_layer

    def get_loss(self, data, chosen = None): # data: (N)
        idx, graphs, labels = data.values()
        pred = self.forward(graphs) # [N]

        # filter nodes
        if chosen is None:
            feat_2 = graphs[-2].x[:,0]
            feat_1 = graphs[-1].x[:,0]
            chosen = (labels!=0) & (feat_1!=0) & (feat_2!=0)
        
        next_pred = pred[chosen]
        next_labels = labels[chosen]
        
        if self.chosen == 'all':
            next_pred = pred
            next_labels = labels

        graphloss = self.mseloss(next_pred,next_labels)
        return graphloss, next_pred.detach().cpu().numpy(), next_labels.detach().cpu().numpy()
    
    def get_embeddings(self, data, chosen):
        idx, graphs, labels = data.values()
        structural_out = torch.cat(
            [self.topological(graphs[t]).x.unsqueeze(1) for t in range(0, self.num_time_steps)],
            dim=1) # [N, T, F]

        temporal_out = self.temporal(structural_out) # [N,T,F]
        embedding = torch.squeeze(temporal_out[:,-1,:]) # [N, F]
        
        return embedding[chosen,:].detach().cpu(), labels[chosen].detach().cpu()
    
