# token embedding is the process of converting tokens in the input to token vectors.
# token ids does not capture the semantic meanings they are simply random numbers
# The embedding captures the semantic meaning.

# using one hot encoding does not work because we will lose the meaning here.

# the embeddings are created randomly at the initialisation. The values are adjusted as
# part of the training process.

# The embedding layer will act same as that of a Linear layer, it is more performant layer compared
# to the Linear layer in neural network. Therefore, in LLM embedding layer is preferred.

import torch

if __name__ == '__main__':
    embedding = torch.nn.Embedding(100, 28)
    print(embedding(torch.tensor([1,2,3,4])))