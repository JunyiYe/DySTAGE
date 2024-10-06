import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import MSELoss
from models.layers import *

class DySTAGE(nn.Module):
    """
    DySTAGE Model for dynamic graph learning with attention mechanisms.

    Parameters:
    -----------
    - args (Namespace): Argument namespace containing hyperparameters for the model.
    - num_nodes (int): Number of nodes/assets in the graph.
    - num_features (int): Number of features per node.
    - edge_scale (float): Number of scales for edge features.
    - valid_feat_idx (Tensor): Indices of valid features for filtering existing nodes.

    Input:
    ------
    - graphs: A sequence of historical graphs, each containing node features, edge indices, shortest paths.

    Output:
    -------
    - prediction (N): predicted excess return for each nodes in the future time step.
    """
    def __init__(self, args, num_nodes, num_features, edge_scale, valid_feat_idx):
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
        self.temporal_drop = args.temporal_drop

        self.structural_attn, self.temporal_attn, self.final = self.build_model()
        self.mseloss = MSELoss()

        self.valid_feat_idx = valid_feat_idx 
            
        
    def forward(self, graphs):
        structural_out = torch.cat(
            [self.structural_attn(graphs[t]).x.unsqueeze(1) for t in range(0, self.num_time_steps)], 
            dim=1) # [N, T, F]
        temporal_out = self.temporal_attn(structural_out) # [N,T,F]
        out = torch.squeeze(self.final(temporal_out[:,-1,:])) # [N]        
        return out


    def build_model(self):
        input_dim = self.num_features

        # 1: Topological Module
        structural_attention_layer = TopologicalLayer(num_nodes = self.num_nodes,
                                                     input_dim=input_dim,
                                                     node_dim = self.structural_node_dim,
                                                     edge_scale = self.edge_scale,
                                                     out_dim=self.temporal_layer_config[0],
                                                     n_heads=self.structural_n_heads,
                                                     num_layers=self.structural_n_layers,
                                                     centrality = self.centrality,
                                                     spatial = self.spatial,
                                                     edge = self.edge)
            
        # 2: Temporal Module
        input_dim = self.temporal_layer_config[0]
        temporal_attention_layers = nn.Sequential()
        for i in range(len(self.temporal_layer_config)):
            layer = TemporalAttentionLayer(input_dim=input_dim,
                                           n_heads=self.temporal_head_config[i],
                                           num_time_steps=self.num_time_steps,
                                           attn_drop=self.temporal_drop,
                                           residual=self.args.residual)
            temporal_attention_layers.add_module(name="temporal_layer_{}".format(i), module=layer)
            input_dim = self.temporal_layer_config[i]
        
        final_layer = nn.Sequential(nn.Linear(input_dim, 1, bias=False),
                                    nn.Tanh())

        return structural_attention_layer, temporal_attention_layers, final_layer


    def get_loss(self, data): # data: (N)
        idx, graphs, labels = data.values()
        pred = self.forward(graphs) # [N]

        # filter existing nodes
        feat_2 = graphs[-2].x[:,self.valid_feat_idx]
        feat_1 = graphs[-1].x[:,self.valid_feat_idx]
        chosen = (labels!=0) & (feat_1!=0) & (feat_2!=0)
        
        next_pred = pred[chosen]
        next_labels = labels[chosen]

        graphloss = self.mseloss(next_pred,next_labels)
        return graphloss, next_pred.detach().cpu().numpy(), next_labels.detach().cpu().numpy()
    
