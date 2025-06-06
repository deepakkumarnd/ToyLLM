import torch
from torch import nn
from transformer import TransformerBlock
from config import GPTConfig
from layer_normalisation import LayerNormalization


class GPTModel(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.emb_dim)
        self.positional_embedding = nn.Embedding(config.context_length, config.emb_dim)
        self.drop_emb = nn.Dropout(config.dropout_rate)

        self.transformers = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.n_layers)]
        )

        self.final_norm = LayerNormalization(config.emb_dim)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, in_idx):
        batch_size, sequence_length = in_idx.shape

        tok_emb = self.token_embedding(in_idx)
        pos_emb = self.positional_embedding(torch.arange(0, sequence_length))

        # build input embedding
        x = tok_emb + pos_emb
        x = self.drop_emb(x)
        x = self.transformers(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


config = GPTConfig(
    emb_dim=3,
    vocab_size=64,
    context_length=4,
    attn_heads=1,
    dropout_rate=0.1,
    n_layers=1,
    qkv_bias=False
)

batches = torch.randint(20, (2, 4))
gpt = GPTModel(config)

print(gpt.forward(batches))
