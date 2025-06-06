import torch
from torch import nn


class SelfAttentionWithTrainableWeightsV1(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(dim_in, dim_out))
        self.W_key = nn.Parameter(torch.rand(dim_in, dim_out))
        self.W_value = nn.Parameter(torch.rand(dim_in, dim_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        attention_score = queries @ keys.T
        attention_weights = torch.softmax(attention_score / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attention_weights @ values
        return context_vec


class SelfAttentionWithTrainableWeightsV2:
    def __int__(self, dim_in, dim_out, qkv_bias=False):
        # the layer is initialised using nn.Linear instead of nn.Parameter
        # bias is set to False so that the bias vector won't be added and this
        # means the operation will be a metrix multiplication.
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_query(x)
        values = self.W_query(x)

        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)

        context_vector = attention_weights @ values
        return context_vector


torch.manual_seed(123)
embedding_dim = 3
input_embedding = torch.rand(6, embedding_dim)
print("Input embeddings")
print(input_embedding)

self_attention = SelfAttentionWithTrainableWeightsV1(embedding_dim, embedding_dim - 1)
context_vector1 = self_attention.forward(input_embedding)
print("Context vector 1")
print(context_vector1)

self_attention = SelfAttentionWithTrainableWeightsV1(embedding_dim, embedding_dim - 1)
context_vector2 = self_attention.forward(input_embedding)
print("Context vector 2")
print(context_vector2)
