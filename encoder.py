import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import MultiHeadAttentionLayer, FeedforwardLayer


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.feedforward = FeedforwardLayer(
            hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # src = [batch size, src len, hid dim]
        # src_mask = [batch size, src len]
        # self attention
        _src, _ = self.self_attention(src, src, src)
        # dropout, residual connection and layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        # positionwise feedforward
        _src = self.feedforward(src)
        # dropout, residual and layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        # src = [batch size, src len, hid dim]
        return src
    
class seesEncoder(nn.Module):
    
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 seq_len,
                 walk_len,
                 n_nodes,
                 device,
                 dropout=0.2,
                 activation=F.relu):
        super(seesEncoder, self).__init__()
        
        self.device = device
        self.in_dim = input_dim
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.walk_len = walk_len
        
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        
        self.node_embedding = nn.Embedding(n_nodes, hid_dim)
        self.expand = nn.Linear(input_dim, hid_dim, bias=True)
        self.s_layers = nn.ModuleList([TransformerEncoderLayer(hid_dim,
                                                             n_heads,
                                                             pf_dim,
                                                             dropout,
                                                             device)
                                     for _ in range(n_layers)])
        
        self.layers = nn.ModuleList([TransformerEncoderLayer(hid_dim,
                                                             n_heads,
                                                             pf_dim,
                                                             dropout,
                                                             device)
                                     for _ in range(n_layers)])
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.pool = torch.nn.parameter.Parameter(torch.rand(walk_len)).to(device)
        
    def forward(self, src, src_mask, walk, walk_src):
        # src, (batch_size, seq_len, input_dim)
        # src_mask, (batch_size, seq_len)
        # walk, (batch_size, walk_len)
        # walk_src, (batch_size, walk_len, seq_len, input_dim)

        src = self.expand(src)
        walk_src = self.expand(walk_src)
        # src, (batch_size, seq_len, hid_dim)
        # walk_src, (batch_size, walk_len, seq_len, hid_dim)
        
        for layer in self.s_layers:
            src = layer(src, src_mask)
            
        src = torch.einsum('ijk,ij->ijk', src, src_mask)
        
        #walk_emb = self.node_embedding(walk) * self.scale
        # walk_emb, (batch_size, walk_len, hid_dim)
        
        #walk_emb = torch.einsum('ijk,j->ik', walk_emb, self.pool)
        #walk_src = torch.einsum('ijkl,j->ikl',walk_src, self.pool)
        # walk_emb, (batch_size, hid_dim)
        # walk_sec, (batch_size, seq_len, hid_dim)
        
        #walk_emb = torch.stack([walk_emb for _ in range(self.seq_len)], dim=1)
        # walk_emb, (batch_size, seq_len, hid_dim)
        
        
        #src = torch.cat((src,walk_emb * walk_src),2)
        #print(src.shape)
        
        #for layer in self.layers:
        #    src = layer(src, src_mask)
            
        return src, src_mask
    
class seesEncoderWithGraphAttention(nn.Module):
    
    def __init__(self,
                 input_dim,
                 hid_dim,
                 n_layers,
                 n_heads,
                 pf_dim,
                 seq_len,
                 walk_len,
                 n_nodes,
                 device,
                 dropout=0.2,
                 activation=F.relu):
        super(seesEncoderWithGraphAttention, self).__init__()
        
        self.device = device
        self.in_dim = input_dim
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.seq_len = seq_len
        self.walk_len = walk_len
        
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        
        self.node_embedding = nn.Embedding(n_nodes+1, hid_dim)
        self.expand = nn.Linear(input_dim, hid_dim, bias=True)
        self.s_layers = nn.ModuleList([TransformerEncoderLayer(hid_dim,
                                                             n_heads,
                                                             pf_dim,
                                                             dropout,
                                                             device)
                                     for _ in range(n_layers)])
        
        self.layers = nn.ModuleList([TransformerEncoderLayer(hid_dim,
                                                             n_heads,
                                                             pf_dim,
                                                             dropout,
                                                             device)
                                     for _ in range(n_layers)])
        
        self.att = nn.Linear(hid_dim * 2, 1, bias=False)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.pool = torch.nn.parameter.Parameter(torch.rand(walk_len)).to(device)
        
    def forward(self, src, src_mask, walk, walk_src, self_node):
        # src, (batch_size, seq_len, input_dim)
        # src_mask, (batch_size, seq_len)
        # walk, (batch_size, walk_len)
        # walk_src, (batch_size, walk_len, seq_len, input_dim)
        # self_node, (batch_size,)

        src = self.expand(src)
        # src, (batch_size, seq_len, hid_dim)
        walk_src = self.expand(walk_src)
        # walk_src, (batch_size, walk_len, seq_len, hid_dim)
        
        for layer in self.s_layers:
            src = layer(src, src_mask)
            
        src = torch.einsum('ijk,ij->ijk', src, src_mask)
        
        walk_emb = self.node_embedding(walk) * self.scale
        # walk_emb, (batch_size, walk_len, hid_dim)
        
        self_emb = self.node_embedding(self_node) * self.scale
        self_emb = torch.stack([self_emb for _ in range(self.walk_len)], dim=1)
        # self_emb, (batch_size, walk_len, hid_dim)
        
        
        walk_att = torch.softmax(self.att(torch.cat((walk_emb, self_emb), 2)).squeeze(2), dim=-1)
        # walk_att, (batch_size, walk_len)
        
        walk_src = torch.einsum('ijkl,ij->ikl',walk_src, walk_att)
        # walk_src, (batch_size, seq_len, input_dim)
        
        
        
        for layer in self.layers:
            walk_src = layer(walk_src, src_mask)
            
        return torch.cat((src, walk_src),2), src_mask
        
        
        
        
        
        
        
        