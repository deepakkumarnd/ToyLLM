import torch
from tokenizer import SimpleTokenizer, build_vocabulary

input_text = "Every journey begins with a single step."

vocab = build_vocabulary(input_text)

context_len = len(vocab)

simple_tokenizer = SimpleTokenizer(vocab)

encodings = simple_tokenizer.encode(input_text)

print("Input -> ", simple_tokenizer.decode(encodings))
print("Encodings -> ", encodings)

torch.manual_seed(0)

word_embeddings = torch.nn.Embedding(context_len, 3)
positional_embedding = torch.nn.Embedding(context_len, 3)

print(word_embeddings.weight)
print("Word vector for encoding id=", encodings[1], ", word=", simple_tokenizer.decode([encodings[1]]), "is", word_embeddings(torch.tensor(encodings[1])))

print("Context length", context_len)
input_embeddings = word_embeddings(torch.tensor(encodings)) + positional_embedding(torch.arange(0, context_len))

print("Word vector for the context", input_embeddings, "dim=", input_embeddings.shape)

attention_score = input_embeddings @ input_embeddings.T

print("Attention score for each token in the context against each other word in the context", attention_score, "dim=", attention_score.shape)

attention_weights = torch.softmax(attention_score, dim=-1)

print("Normalised attention weights for each token in the context vector against each other token", attention_weights, "dim=", attention_weights.shape)

# scaling input embedding with attention weights
# attention_weights * input_embeddings

context_embedding = attention_weights @ input_embeddings

print("Context embedding for the input embeddings", context_embedding)



