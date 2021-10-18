import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import MultiHeadAttentionLayer, FeedforwardLayer


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 hid_dim,
                 n_heads,
                 pf_dim,
                 dropout,
                 device):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(
            hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(
            hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self,
                trg,
                enc_src,
                trg_mask,
                src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]
        # self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        # dropout, residual connection and layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # encoder attention
        _trg, attention = self.encoder_attention(
            trg, enc_src, enc_src, src_mask)
        # dropout, residual connection and layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # positionwise feedforward
        _trg = self.positionwise_feedforward(trg)
        # dropout, residual and layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        return trg, attention
    
class seesDecoder(nn.Module):
    
    def __init__(self,
                 hid_dim,
                 out_dim,
                 seq_len,
                 device,
                 dropout=0.2,
                 activation=F.relu):
        super(seesDecoder, self).__init__()
        
        self.device = device
        self.hid_dim = hid_dim
        
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        
        self.FC = nn.Linear(hid_dim * seq_len, out_dim, bias=True)

        
    def forward(self, src):
        # src, (batch_size, seq_len, hid_dim)
        src = src.view(src.shape[0], src.shape[1]*src.shape[2])
        # src, (batch_size, seq_len * hid_dim)
        
        src = self.FC(src)
        # src, (batch_size, out_dim)
        
        return src

    
class seesDecoderWithAttInputs(nn.Module):
    
    def __init__(self,
                 hid_dim,
                 out_dim,
                 seq_len,
                 device,
                 dropout=0.2,
                 activation=F.relu):
        super(seesDecoderWithAttInputs, self).__init__()
        
        self.device = device
        self.hid_dim = hid_dim
        
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        
        self.FC1 = nn.Linear(hid_dim * 2, hid_dim, bias=True)
        self.FC2 = nn.Linear(hid_dim * seq_len, out_dim, bias=True)

        
    def forward(self, src):
        # src, (batch_size, seq_len, hid_dim * 2)
        src = self.FC1(src)
        # src, (batch_size, seq_len, hid_dim)
        
        src = src.view(src.shape[0], src.shape[1]*src.shape[2])
        # src, (batch_size, seq_len * hid_dim)
        
        src = self.FC2(src)
        # src, (batch_size, out_dim)
        
        return src
    
class seesDecoderWithMultiOut(nn.Module):
    
    def __init__(self,
                 hid_dim,
                 out_dim,
                 seq_len,
                 device,
                 dropout=0.2,
                 activation=F.relu):
        super(seesDecoderWithMultiOut, self).__init__()
        
        self.device = device
        self.hid_dim = hid_dim
        self.seq_len = seq_len
        
        self.dropout = nn.Dropout(p=dropout)
        self.activation = activation
        
        self.FC = nn.Linear(hid_dim * 2, hid_dim, bias=True)
        self.futureFC = nn.ModuleList(nn.Linear(hid_dim * seq_len,
                                                    out_dim,
                                                    bias=True) for _ in range(seq_len))

        
    def forward(self, src):
        # src, (batch_size, seq_len, hid_dim * 2)
        src = self.FC(src)
        # src, (batch_size, seq_len, hid_dim)
        
        src = src.view(src.shape[0], src.shape[1]*src.shape[2])
        # src, (batch_size, seq_len * hid_dim)
        
        out = []
        for layer in self.futureFC:
            out.append(layer(src))
        
        out = torch.cat(out, dim=1)
        # out, (batch_size, seq_len)
        
        return out