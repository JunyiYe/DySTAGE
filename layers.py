import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import softmax, degree
import copy


class TopologicalLayer(nn.Module):
    def __init__(self, 
                num_nodes,
                input_dim, 
                node_dim,
                edge_scale,
                out_dim,
                n_heads, 
                num_layers,
                centrality = True,
                spatial = True,
                edge = True
                ):
        super(TopologicalLayer, self).__init__()
        self.input_dim = input_dim
        self.node_dim = node_dim
        self.edge_scale = edge_scale
        self.num_nodes = num_nodes
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.out_dim = out_dim

        self.isCentrality = centrality
        self.isSpatial = spatial
        self.isEdge = edge
        
        self.node_linear = nn.Linear(input_dim, node_dim)

        if self.isCentrality:
            self.centrality_encoding = CentralityEncoding(num_nodes = num_nodes, node_dim=node_dim)
        if self.isSpatial:
            self.spatial_encoding = SpatialEncoding(num_nodes = num_nodes, node_dim = node_dim)
        if self.isEdge:
            self.edge_encoding = EdgeEncoding(scale = self.edge_scale)

        self.layers = nn.ModuleList([
            MultiHeadAttention(              
                node_dim=self.node_dim,
                n_heads=self.n_heads) for _ in range(self.num_layers)
        ])
        self.out_linear = nn.Linear(self.node_dim, self.out_dim)
        

    def forward(self, graph):
        graph = copy.deepcopy(graph) 
        x = graph.x # (N, F)
        edge_index = graph.edge_index # (2, #edges)
        shortest_path_len = graph.shortest_path_len.long() # (N, N)
        edge_feat = graph.edge_feat # (n,n,scale)
        mask_idx = (torch.sum(x, axis=1)!=0).float()
        
        x = self.node_linear(x) # (N, node_dim)
        if self.isCentrality:         
            x = self.centrality_encoding(x, edge_index)   # (N, node_dim)
        
        spatial_embeddings = self.spatial_encoding(shortest_path_len) if self.isSpatial else None # (N, N, node_dim)
        edge_embeddings = self.edge_encoding(edge_feat) if self.isEdge else None
        for layer in self.layers:
            x = layer(x, mask_idx, spatial_embeddings, edge_embeddings) # (N, n_heads * dim)

        graph.x = self.out_linear(x)
        return graph

    
    
class CentralityEncoding(nn.Module):
    def __init__(self, num_nodes, node_dim):
        super().__init__() 
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.degree_encoding = nn.Embedding(self.num_nodes, self.node_dim)

    def forward(self, x, edge_index):
        node_degree = degree(index=edge_index[1], num_nodes=self.num_nodes).long()
        x += self.degree_encoding(node_degree)
        return x
    

class SpatialEncoding(nn.Module):
    def __init__(self, num_nodes, node_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.spatial_embeddings = nn.Embedding(11, 1)

    def forward(self, shortest_path_len):
        spatial_matrix = self.spatial_embeddings(shortest_path_len).squeeze() # (N, N)
        return spatial_matrix


class EdgeEncoding(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.linear = nn.Linear(scale, scale)

    def forward(self, edge_feat):
        edge_matrix = self.linear(edge_feat).mean(dim=2).squeeze()
        return edge_matrix
    

class MultiHeadAttention(nn.Module):
    def __init__(self, node_dim, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(node_dim)
        self.heads = nn.ModuleList(
            [AttentionHead(node_dim, node_dim, node_dim) for _ in range(n_heads)]
        )
        self.linear = nn.Linear(n_heads * node_dim, node_dim)
        self.ln2 = nn.LayerNorm(node_dim)
        self.ff = nn.Linear(node_dim, node_dim)

    def forward(self, x, mask_idx, spatial, edge):
        x_ln = self.ln1(x) # (N, node_dim)
        x_out =  self.linear(
            torch.cat([
                attention_head(x_ln, x_ln, x_ln, mask_idx, spatial, edge) for attention_head in self.heads
            ], dim=-1)
        ) + x # (N, node_dim) 
        x_attn = self.ff(self.ln2(x_out)) + x_out
        return x_attn


class AttentionHead(nn.Module):
    def __init__(self, dim_in, dim_q, dim_k):
        super().__init__()
        self.dim_in = dim_in,
        self.dim_q = dim_q,
        self.q = nn.Linear(dim_in, dim_q)
        self.k = nn.Linear(dim_in, dim_k)
        self.v = nn.Linear(dim_in, dim_k)

    def forward(self, query, key, value, mask_idx, spatial, edge):
        query = self.q(query)
        key = self.k(key)
        value = self.v(value) # (N, node_dim)
        a = query.mm(key.transpose(0, 1)) / query.size(-1) ** 0.5 #(N,N)
        if spatial is not None:
            a += spatial
        if edge is not None:
            a += edge

        mask = torch.matmul(mask_idx.reshape(-1,1), (mask_idx.reshape(1,-1)))
        a_mask = a.masked_fill(mask==0, -2**32+1) # for nonexisting node, attention =0
        softmax = torch.softmax(a_mask, dim=-1).masked_fill(mask==0, 0) # convert nan into 0
        x = softmax.mm(value)
        return x

        
class TemporalLayer(nn.Module):
    def __init__(self, 
                input_dim, 
                n_heads, 
                num_time_steps, 
                attn_drop, 
                residual):
        super(TemporalLayer, self).__init__()
        self.n_heads = n_heads
        self.num_time_steps = num_time_steps
        self.residual = residual

        self.position_embeddings = nn.Parameter(torch.Tensor(num_time_steps, input_dim))
        self.Q_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.K_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.V_embedding_weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        self.lin = nn.Linear(input_dim, input_dim, bias=True)
        self.attn_dp = nn.Dropout(attn_drop)
        self.xavier_init()


    def forward(self, inputs):
        # Add position embeddings to input
        position_inputs = torch.arange(0,self.num_time_steps).reshape(1, -1).repeat(inputs.shape[0], 1).long().to(inputs.device)
        temporal_inputs = inputs + self.position_embeddings[position_inputs] # [N, T, F]

        q = torch.tensordot(temporal_inputs, self.Q_embedding_weights, dims=([2],[0])) # [N, T, F]
        k = torch.tensordot(temporal_inputs, self.K_embedding_weights, dims=([2],[0])) # [N, T, F]
        v = torch.tensordot(temporal_inputs, self.V_embedding_weights, dims=([2],[0])) # [N, T, F]

        split_size = int(q.shape[-1]/self.n_heads)
        q_ = torch.cat(torch.split(q, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        k_ = torch.cat(torch.split(k, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        v_ = torch.cat(torch.split(v, split_size_or_sections=split_size, dim=2), dim=0) # [hN, T, F/h]
        
        outputs = torch.matmul(q_, k_.permute(0,2,1)) # [hN, T, T]
        outputs = outputs / (self.num_time_steps ** 0.5)

        diag_val = torch.ones_like(outputs[0])
        tril = torch.tril(diag_val)
        masks = tril[None, :, :].repeat(outputs.shape[0], 1, 1) # [h*N, T, T]
        padding = torch.ones_like(masks) * (-2**32+1)
        outputs = torch.where(masks==0, padding, outputs)
        outputs = F.softmax(outputs, dim=2)
        self.attn_wts_all = outputs # [h*N, T, T]

        if self.training:
            outputs = self.attn_dp(outputs)
        outputs = torch.matmul(outputs, v_)  # [hN, T, F/h]
        outputs = torch.cat(torch.split(outputs, split_size_or_sections=int(outputs.shape[0]/self.n_heads), dim=0), dim=2) # [N, T, F]

        outputs = self.feedforward(outputs)
        if self.residual:
            outputs = outputs + temporal_inputs
        return outputs

    def feedforward(self, inputs):
        outputs = F.relu(self.lin(inputs))
        return outputs + inputs


    def xavier_init(self):
        nn.init.xavier_uniform_(self.position_embeddings)
        nn.init.xavier_uniform_(self.Q_embedding_weights)
        nn.init.xavier_uniform_(self.K_embedding_weights)
        nn.init.xavier_uniform_(self.V_embedding_weights)
