import torch


class SelfAttentionWithTrainableWeightsV1(torch.nn.Module):
    def __init__(self, dim_in, dim_out, qkv_bias=False):
        super().__init__()
        # self.query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_query = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(dim_in, dim_out, bias=qkv_bias)

    def forward(self, x):
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        attention_score = queries @ keys.T
        attention_weights = torch.softmax(attention_score / keys.shape[-1] ** 0.5, dim=-1)

        context_vec = attention_weights @ values
        return context_vec

torch.manual_seed(0)
embedding_dim = 3
input_embedding = torch.rand(4, embedding_dim)
print(input_embedding)

self_attention = SelfAttentionWithTrainableWeightsV1(embedding_dim, embedding_dim-1)
context_vector = self_attention.forward(input_embedding)
print(context_vector)

