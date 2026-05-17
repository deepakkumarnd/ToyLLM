import torch
from torch import nn
from transformers import GPT2Config, GPT2Model

from namelm.config import LLMConfig


class NameModel(nn.Module):
    def __init__(self, config: LLMConfig):
        super().__init__()

        hf_config = GPT2Config(
            vocab_size=config.vocab_size,
            n_embd=config.emb_dim,
            n_layer=config.n_layers,
            n_head=config.attn_heads,
            n_positions=config.context_length,
            n_ctx=config.context_length,
            attn_pdrop=config.dropout_rate,
            embd_pdrop=config.dropout_rate,
            resid_pdrop=config.dropout_rate,
            add_cross_attention=False,
        )

        self.transformer = GPT2Model(hf_config)
        self.out_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

    def forward(self, in_idx: torch.Tensor) -> torch.Tensor:
        x = self.transformer(in_idx).last_hidden_state
        return self.out_head(x)


if __name__ == "__main__":
    config = LLMConfig()
    model = NameModel(config)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    batch = torch.randint(0, config.vocab_size, (2, config.context_length))
    logits = model(batch)
    print(f"Input  shape: {batch.shape}")
    print(f"Output shape: {logits.shape}")  # (batch, context_length, vocab_size)
