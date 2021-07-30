import torch
from torch.nn import functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel
torch.set_grad_enabled(False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = GPT2Tokenizer.from_pretrained('./gpt2')
model = GPT2LMHeadModel.from_pretrained('./gpt2').eval()
model = model.to(device)


def extend(text, size=20):
    if not text:
        text = '<|endoftext|>'
    tokens = tokenizer.encode(text)
    tokens = torch.tensor([tokens]).to(device)
    tokens = model.generate(tokens, max_length=size+tokens.shape[1], do_sample=True)
    tokens = tokens[0].tolist()
    return tokenizer.decode(tokens)


if __name__ == "__main__":
    test_text = 'Microsoft and Google'
    extended = extend(test_text, 25)
    print(extended)
