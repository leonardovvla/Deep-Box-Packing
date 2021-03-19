"""

Deep Box Packing - Mapping Transformer: Neural Network architecture used by DBP

Author: Leonardo Albuquerque - ETH Zürich, 2021

Based on Aladdin Persson's implementation of the Transformer network available at: 
https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/transformer_from_scratch

Unless stated otherwise by an "A.P." signature, comments by Leonardo Albuquerque

"""

# ==================================================================================================== #
# ============================================== IMPORTS ============================================= #
# ==================================================================================================== #

import torch
import torch.nn as nn

import numpy as np

import matplotlib.pyplot as plt
import matplotlib

from torchvision import utils

# ==================================================================================================== #
# ========================================== NETWORK MODULES ========================================= #
# ==================================================================================================== #
#                                                                                                      #
#                                                Packer                                                #
#                                                  |                                                   #
#                                          Mapping Transformer                                         #
#                                          |                 |                                         #
#                                    Encoder              Decoder                                      #
#                                       |                 |     |                                      #
#                         TransformerBlock     DecoderBlock     MapAttention                           #
#                                |                  |           (Policy Head)                          #
#                           SelfAttention    TransformerBlock   & Value head                           #
#                                                   |                                                  #
#                                             SelfAttention                                            #
#                                                                                                      #
# ==================================================================================================== #

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, query, keys, values):
        # A.P.: Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        #A.P.: Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # A.P.: (N, value_len, heads, head_dim)
        keys = self.keys(keys)        # A.P.: (N, key_len, heads, head_dim)
        queries = self.queries(query) # A.P.: (N, query_len, heads, heads_dim)

        # A.P.: Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # A.P.: queries shape: (N, query_len, heads, heads_dim),
        # A.P.: keys shape: (N, key_len, heads, heads_dim)
        # A.P.: energy: (N, heads, query_len, key_len)

        # A.P.: Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # A.P.: attention shape: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # A.P.: attention shape: (N, heads, query_len, key_len)
        # A.P.: values shape: (N, value_len, heads, heads_dim)
        # A.P.: out after matrix multiply: (N, query_len, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.fc_out(out)
        # A.P.: Linear layer doesn't modify the shape, final shape will be (N, query_len, embed_size)

        return out

class MapAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(MapAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.h1_out = nn.Linear(heads, 1)

    def forward(self, query, keys, values):
        # A.P.: Get number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # A.P.: Split the embedding into self.heads different pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        query = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)  # A.P.: (N, value_len, heads, head_dim)
        keys = self.keys(keys)        # A.P.: (N, key_len, heads, head_dim)
        queries = self.queries(query) # A.P.: (N, query_len, heads, heads_dim)

        # A.P.: Einsum does matrix mult. for query*keys for each training example
        # with every other training example, don't be confused by einsum
        # it's just how I like doing matrix multiplication & bmm

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        # A.P.: queries shape: (N, query_len, heads, heads_dim),
        # A.P.: keys shape: (N, key_len, heads, heads_dim)
        # A.P.: energy: (N, heads, query_len, key_len)

        # A.P.: Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention = energy / (self.embed_size ** (1 / 2))
        # A.P.: attention shape: (N, heads, query_len, key_len)

        attention = attention.permute(0,2,3,1)
        # attention shape: (N, query_len, key_len, heads)

        # Combine attention heads
        out = (self.h1_out(attention)).reshape(
            N, query_len * key_len
        )

        return out    

class ValueHead(nn.Module):
    def __init__(self, embed_size, container_area, dropout):
        super(ValueHead, self).__init__()

        self.combine_embs = nn.Linear(embed_size, 1)
        self.combine_cells = nn.Linear(container_area, 1)

    def forward(self, x):
        
        x = self.combine_embs(x)
        x_t = x.transpose(2,1)
        out = self.combine_cells(x_t)
        
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attention = self.attention(query, key, value)

        # A.P.: Add skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        box_max_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
    ):

        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.box_embedding = nn.Linear(3, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        out = self.box_embedding(x)
        out = self.dropout(out)

        i=0
        # A.P.: In the Encoder the query, key, value are all the same, it's in the
        # decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out)
            i=i+1

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )

    def forward(self, x, key, value):
        query = x 
        out = self.transformer_block(query, key, value)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        container_max_height,
        container_area,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
    ):
        super(Decoder, self).__init__()
        self.device = device

        self.container_embedding = nn.Embedding(container_max_height+1, embed_size)

        self.container_conv1 = nn.Conv2d(in_channels=embed_size,out_channels=3*embed_size,kernel_size=7, padding=(3,3))
        self.container_conv2 = nn.Conv2d(in_channels=3*embed_size,out_channels=2*embed_size,kernel_size=5, padding=(2,2))
        self.container_conv3 = nn.Conv2d(in_channels=2*embed_size,out_channels=embed_size,kernel_size=3, padding=(1,1))

        self.lin_map = nn.Linear(16,100)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.selfAttention = SelfAttention(embed_size, heads)

        self.mapAttention = MapAttention(embed_size, heads)

        self.valueHead = ValueHead(embed_size, container_area, dropout)

        self.dropout = nn.Dropout(dropout)

    def embedContainer(self, x):

        x = self.container_embedding(x.type(torch.LongTensor)).permute(0,3,1,2)   # Generate embed_size dimensions per grid cell
        x = self.container_conv1(x)                                               # Apply convolutional series to embed in each cell
        x = self.container_conv2(x)                                               # information of its neightborhood  
        x = self.container_conv3(x).permute(0,2,3,1)                             

        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])            # Flatten convolved heightmap
        # x = self.selfAttention(x, x, x)                                         # Have each cell attend to all others (not used)
        x = self.dropout(x)

        return x

    def forward(self, x, enc_out):

        x = self.embedContainer(x)

        i=0;

        for layer in self.layers:
            x = layer(x, enc_out, enc_out)
            i=i+1

        out_action = self.mapAttention(x, enc_out, enc_out)  # out_action = 1 flattenned 6*n*W*L att map per container (N)

        out_value = self.valueHead(x)                        # out_value = 1 scalar value per container (N)

        return [out_action, out_value]


class MappingTransformer(nn.Module):
    def __init__(
        self,
        box_max_size,
        container_max_size,
        container_area,
        embed_size=32,
        num_layers=6,
        forward_expansion=4,
        heads=4,
        dropout=0,
        device="cpu",
    ):

        super(MappingTransformer, self).__init__()

        self.encoder = Encoder(
            box_max_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
        )

        self.decoder = Decoder(
            container_max_size,
            container_area,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
        )

        self.device = device

    def forward(self, src, trg):

        enc_src = self.encoder(src)

        out_action, out_value = self.decoder(trg, enc_src)

        return [out_action, out_value]

# ==================================================================================================== #
# =========================================== MPT WRAPPER ============================================ #
# ==================================================================================================== #

class Packer(nn.Module):
    def __init__(
        self,
        box_max_size=10,
        container_max_size=101,
        container_area=100,
        embed_size=32,
        num_layers=6,
        forward_expansion=4,
        heads=6,
        device="cpu",
    ):

        super(Packer, self).__init__()

        self.mpt = MappingTransformer(box_max_size, container_max_size, container_area, embed_size=embed_size, num_layers=num_layers, forward_expansion=forward_expansion, heads=heads, device=device).to(device)

    def forward(self, box_info, container_hm):

        box_info = torch.transpose(box_info,1,2)

        container_hm = container_hm.reshape(container_hm.shape[0], container_hm.shape[2], container_hm.shape[3])

        logits, values = self.mpt(box_info[:,:,1:], container_hm)

        return [logits, values]
