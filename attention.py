import torch
from torch import nn


# AKA masked attention, causal attention prevents model from accessing future tokens
# in the input sequence.
class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out, dropout, context_length, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        # dropout will be used only during the training phase
        self.dropout = nn.Dropout(dropout)
        # these are not part of model parameters and are not part of training
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, batch):
        batch_size, num_tokens, d_in = batch.shape
        queries = self.W_query(batch)
        keys = self.W_key(batch)
        values = self.W_value(batch)

        attention_score = queries @ keys.transpose(1, 2)
        attention_score = attention_score.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attention_weights = torch.softmax(attention_score / keys.shape[-1] ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vec = attention_weights @ values
        return context_vec


class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, dim_in, dim_out, n_heads, dropout, context_length, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(dim_in, dim_out, dropout, context_length, qkv_bias) for _ in range(n_heads)]
        )

    def forward(self, batch):
        return torch.concat([head(batch) for head in self.heads], dim=-1)


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, dropout, n_heads, context_length, qkv_bias=False):
        super().__init__()
        assert (dim_out % n_heads) == 0, "dim_out must be a multiple of n_heads"

        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))
        self.dropout = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.head_dim = dim_out // n_heads
        self.dim_out = dim_out
        self.out_projection = nn.Linear(dim_out, dim_out)  # This layer is setting the bias

    def forward(self, batch):
        b, num_tokens, d_in = batch.shape

        queries = self.W_query(batch)  # shape = b x num_tokens x d_out
        keys = self.W_key(batch)
        values = self.W_value(batch)

        # reshape queries by splitting d_out, keys and values by treating them as a
        # concatenated values from multiple heads, here we split the last dimension
        # into n_heads x head_dim

        queries = queries.view(b, num_tokens, self.n_heads, self.head_dim)
        keys = keys.view(b, num_tokens, self.n_heads, self.head_dim)
        values = values.view(b, num_tokens, self.n_heads, self.head_dim)

        # next we need to group the results from the same heads by
        # transposing number_tokens and n_heads
        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)  # b x n_heads x num_tokens x head_dim

        attention_scores = queries @ keys.transpose(2, 3)  # b x n_heads x num_tokens x num_tokens

        # masked scores
        masked_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attention_scores = attention_scores.masked_fill(masked_bool, -torch.inf)
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 2, dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vector = attention_weights @ values  # b x n_heads x num_tokens x head_dim
        # each token having results from multiple heads
        context_vector = context_vector.transpose(1, 2)  # b x num_tokens x n_heads x head_dim
        # concatenate the different heads results
        context_vector = context_vector.contiguous().view(b, num_tokens, self.dim_out)
        context_vector = self.out_projection(context_vector)

        return context_vector


# inputs = torch.rand(6, 3)

# batch = torch.stack([inputs, inputs], dim=0)

# attn = CausalAttention(3, 2, context_length=6, dropout=0.1)
# context_vector1 = attn.forward(batch)
# print("Context vector1")
# print(context_vector1)

# multi_attn = MultiHeadAttentionWrapper(3, 2, 2, dropout=0.1, context_length=6)
# context_vector1 = multi_attn(batch)
# print(context_vector1)
#
# attn2 = MaskedMultiHeadAttention(3, 6, 0.1, 3, 6)
#
# context_vector2 = attn2(batch)
# print(context_vector2)