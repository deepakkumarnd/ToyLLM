class GPTConfig:
    def __init__(self,
                 emb_dim=768,
                 vocab_size=52000,
                 context_length=4,
                 attn_heads=12,
                 dropout_rate=0.1,
                 n_layers=10,
                 qkv_bias = False):

        self.emb_dim = emb_dim
        self.context_length = context_length
        self.attn_heads = attn_heads
        self.dropout_rate = dropout_rate
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.qkv_bias = qkv_bias

    def __str__(self):
        configuration = {
            "emb_dim": self.emb_dim,
            "vocab_size": self.vocab_size,
            "context_length": self.context_length,
            "attn_heads": self.attn_heads,
            "dropout_rate": self.dropout_rate,
            "n_layers": self.n_layers,
            "qkv_bias": self.qkv_bias
        }

        return str(configuration)


# config = GPTConfig(dropout_rate=0.8)
#
# print(config)
