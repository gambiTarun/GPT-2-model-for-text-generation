import torch

from src.model import GPT, GPTConfig
from train import MyTrainDataset, num_embd, num_head, num_layer, block_size, bias, dropout

max_num_tokens = 2048
_,_,decode,vocab_size = MyTrainDataset(1)

gptconf = GPTConfig(
        num_layer=num_layer, 
        num_head=num_head, 
        num_embd=num_embd, 
        block_size=block_size,
        bias=bias, 
        vocab_size=vocab_size,
        dropout=dropout)
model = GPT(gptconf)

snapshot = torch.load('snapshot.pth', map_location='mps' if torch.backends.mps.is_available() else 'cpu')
model.load_state_dict(snapshot["model"])

gen = model.generate(idx = torch.zeros((1,1), dtype=torch.long), max_len=max_num_tokens, temperature=1.0, top_k=None)[0].tolist()
more = decode(gen)

print(gen, more)
with open('output.txt', 'w') as f:
        f.write(more)