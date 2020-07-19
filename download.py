# Downloads model(s) during docker build
from transformers import GPT2LMHeadModel, GPT2Tokenizer

GPT2LMHeadModel.from_pretrained('gpt2').save_pretrained('./gpt2')
GPT2Tokenizer.from_pretrained('gpt2').save_pretrained('./gpt2')

