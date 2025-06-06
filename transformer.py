from torch import nn

from attention import MaskedMultiHeadAttention
from config import GPTConfig
from feed_forward import FeedForward
from layer_normalisation import LayerNormalization


class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attention = MaskedMultiHeadAttention(
            dim_in=config.emb_dim,
            dim_out=config.emb_dim,
            dropout=config.dropout_rate,
            n_heads=config.attn_heads,
            context_length=config.context_length,
            qkv_bias=config.qkv_bias
        )
        self.feed_forward_network = FeedForward(config)
        self.norm1 = LayerNormalization(config.emb_dim)
        self.norm2 = LayerNormalization(config.emb_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.attention(x)
        x = self.dropout(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.feed_forward_network(x)
        x = self.dropout(x)
        x = x + shortcut

        return x


# config = GPTConfig(
#     emb_dim=768,
#     vocab_size=52000,
#     context_length=4,
#     attn_heads=16,
#     dropout_rate=0.1,
#     n_layers=10,
#     qkv_bias=False
# )

# trf = TransformerBlock(config)