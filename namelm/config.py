from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    vocab_size: int = 10000
    emb_dim: int = 256
    context_length: int = 8
    attn_heads: int = 8
    n_layers: int = 4
    dropout_rate: float = 0.1
    qkv_bias: bool = False
