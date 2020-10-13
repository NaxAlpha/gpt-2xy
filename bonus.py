# Simple script to fine-tune GPT-2 on custom dataset

import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer

is_gpu = True
model_name = 'gpt2'

device = 'cuda' if torch.cuda.is_available() and is_gpu else 'cpu'


class SingleTextFile(Dataset):
    def __init__(self, fn, max_len=128):
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        with open(fn) as f:
            text = f.read()
        self.tokens = tokenizer.encode(text)
        self.max_len = max_len
    
    def __len__(self):
        return len(self.tokens)-self.max_len
    
    def __getitem__(self, i):
        block = self.tokens[i:i+self.max_len]
        return torch.tensor(block)


if __name__ == '__main__':
  ds = SingleTextFile('big.txt', 128)
  dl = DataLoader(ds, batch_size=8, shuffle=True)
  
  model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
  optim = torch.optim.Adam(model.parameters(), lr=1e-6)
  
  prog = tqdm(dl)
  for i, batch in enumerate(prog):
      batch = batch.cuda()
      loss, *_ = model(batch, labels=batch)

      optim.zero_grad()
      loss.backward()
      optim.step()
      
  torch.save(model.state_dict(), 'gpt2.pt')
