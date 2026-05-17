from pathlib import Path

import torch
from datasets import Dataset

from hf.tokenizer import name_tokenizer
from namelm.config import LLMConfig

_DEFAULT_NAMES = Path(__file__).parent.parent / "data" / "indian-names.txt"


def _build_dataset(path: str | Path, config: LLMConfig) -> Dataset:
    tok = name_tokenizer()
    sep_id = tok.token_to_id("[SEP]")

    # tokenize every name and join with [SEP] into one long token stream
    token_stream = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                token_stream.extend(tok.encode(name).ids)
                token_stream.append(sep_id)

    # slice into (input, target) windows of context_length
    inputs, targets = [], []
    for i in range(0, len(token_stream) - config.context_length):
        inputs.append(token_stream[i : i + config.context_length])
        targets.append(token_stream[i + 1 : i + 1 + config.context_length])

    return Dataset.from_dict({"input_ids": inputs, "target_ids": targets})


def create_dataloader(
    config: LLMConfig,
    path: str | Path = _DEFAULT_NAMES,
    batch_size: int = 32,
    shuffle: bool = True,
) -> torch.utils.data.DataLoader:
    ds = _build_dataset(path, config)
    ds.set_format(type="torch", columns=["input_ids", "target_ids"])
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def create_train_val_dataloaders(
    config: LLMConfig,
    path: str | Path = _DEFAULT_NAMES,
    val_split: float = 0.1,
    batch_size: int = 32,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    ds = _build_dataset(path, config)
    split = ds.train_test_split(test_size=val_split, seed=42)
    for s in split.values():
        s.set_format(type="torch", columns=["input_ids", "target_ids"])
    train_loader = torch.utils.data.DataLoader(split["train"], batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(split["test"], batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


if __name__ == "__main__":
    config = LLMConfig()
    loader = create_dataloader(config, batch_size=8, shuffle=False)
    inputs, targets = next(iter(loader))["input_ids"], next(iter(loader))["target_ids"]
    print(f"Dataset size : {len(loader.dataset)}")
    print(f"Batches      : {len(loader)}")
    print(f"Input shape  : {inputs.shape}")
    print(f"Target shape : {targets.shape}")
    print(f"Input batch  :\n{inputs}")
    print(f"Target batch :\n{targets}")
