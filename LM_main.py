import math
import torch
from pytorch_transformers import GPT2Tokenizer, GPT2LMHeadModel


# Load pre-trained model (weights)
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

# Load pre-trained model tokenizer (vocabulary)
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def perplexity(sentence):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    loss = model(tensor_input, labels=tensor_input)
    return math.exp(loss[0].item() / len(tokenize_input))


if __name__ == '__main__':
    a = ["i wrote a book, i wrote a book, i wrote a book, i wrote a book,i wrote a book, i wrote a book.",
         "i wrote a book.",
         "i wrote a book about the life of two young people who fall in love with each other."]

    print([perplexity(i) for i in a])
