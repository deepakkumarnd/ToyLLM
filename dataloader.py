from torch.utils.data import Dataset, DataLoader
import tiktoken
import torch

# Tokenization and data loading

with open("hp1.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


# raw_text = """
# Harry Potter and the Sorcerer's Stone
#
#
# CHAPTER ONE
#
# THE BOY WHO LIVED
#
# Mr. and Mrs. Dursley, of number four, Privet Drive, were proud to say
# that they were perfectly normal, thank you very much. They were the last
# people you'd expect to be involved in anything strange or mysterious,
# because they just didn't hold with such nonsense.
# """


class GPTDatasetV1(Dataset):
    def __init__(self, text, tokenizer, context_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})

        for i in range(0, len(token_ids) - context_length, stride):
            input_chunk = token_ids[i:i + context_length]
            output_chunk = token_ids[i + 1:i + 1 + context_length]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(output_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# a batch size is used to build a batch containing few inputs identified by batch_size. The batch size
# means that the weights will be updated after processing that many inputs, using a smaller batch size may
# increase the noise in the training.

# context length refers to the number of tokens in a single input.

# stride refers to the window size by which we shift while processing each input, A when context size is
# equal to or less than stride size then the batch will not have any overlapping inputs. The overlap may
# lead to overfitting and that is not recommended while training.

# every input

def create_dataloader_v1(text, batch_size=4, context_length=256, stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(text, tokenizer, context_length, stride)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                             num_workers=num_workers)
    return data_loader


if __name__ == '__main__':
    dataloader = create_dataloader_v1(raw_text, batch_size=8, context_length=4, stride=4, shuffle=False, num_workers=1)

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    input, labels = first_batch

    print("Input", input)
    print("Label", labels)
    # second_batch = next(data_iter)
    # input, labels = second_batch
    # print("Input", input)
    # # print("Label", labels)
