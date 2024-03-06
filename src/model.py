import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class GPTParameter():
    block_size: int = 8
    vocab_size: int = 50257
    num_layer: int = 12
    num_head: int = 12
    num_embd: int = 768
    dropout: float = 0.0
    bias: bool = False # use bias in linear layers and layer norm. no bias is little faster

class LayerNormalization(nn.Module):
    """ 
    Layer Normalization with optional bias parameter.
    """
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        
    def forward(self, x):
        return nn.functional.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)
    
class CausalSelfAttention(nn.Module):
    """ 
    Causal Self attention module is used for the decoder stack. 
    In the encoder stack, non-causal and cross-attention is used.
    Causal implies that the attention is applied to the past and not the future.
    """
    def __init__(self, config):
        super().__init__()
        # input head k,v,q parameters all batched into one layer
        self.c_attn = nn.Linear(config.num_embd, 3 * config.num_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(3 * config.num_embd, config.num_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.num_head
        self.n_embd = config.num_embd
        self.flash = hasattr(self, 'scaled_dot_product_attention')
        if not self.flash:
            print("Using slow attention. Flash attention requires PyTorch >= 2.0")
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
            
    def forward(self, x):
        # batch, time (sequence length or context window), channel (embedding size or num_embd)
        B, T, C = x.shape
        query, key, value = self.c_attn(x).chunk(3, dim=-1)         # TODO: check if this is correct
        query = query.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, num_Head, T, Head_size)
        key = key.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)      # (B, num_Head, T, Head_size)
        value = value.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, num_Head, T, Head_size)
        
        if self.flash:
            # use fast attention, requires PyTorch >= 2.0
            # no custom attn_mask is used on top of the standard causal mask
            # causal implies that the attention is applied to the past and not the future
            w = F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout=self.dropout if self.training else 0, is_causal=True)
        else:
            attn = query @ key.transpose(-2, -1) * (1 / math.sqrt(key.size(-1))) 
            attn = attn.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            attn = attn.softmax(dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ value # (1, 1, T, T) @ (B, nH, T, Hs) -> (B, nH, T, Hs)
            
        # reassemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, nH, Hs) -> (B, T, C)
        # output projection
        y = self.c_proj(y)
        return self.resid_dropout(y)
    
class MLP(nn.Module):
    """ 
    MLP module for the decoder stack.
    This is the layer where the fetched embeddings from head talk to each other.
    """
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.num_embd, 4 * config.num_embd, bias=config.bias) # the first linear layer
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.num_embd, config.num_embd, bias=config.bias) # the second linear layer
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x):
        x = self.c_fc(x)
        x = self.c_proj(self.gelu(x))
        return self.dropout(x)
    
class Block(nn.Module):
    """
    Transformer block for the decoder stack.
    This block contains the residual skip connections and will be repeated for num_layer times.
    """
    def __init__(self, config):
        super().__init__()
        self.ln1 = LayerNormalization(config.num_embd, config.bias)
        self.attn = CausalSelfAttention(config) 
        self.ln2 = LayerNormalization(config.num_embd, config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    
class GPT(nn.Module):
    """
    This is the main GPT model where everything is put together.
    """
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None, "config.vocab_size must be specified and > 0"
        assert config.block_size is not None, "config.block_size must be specified and > 0"

        self.transformer = nn.Sequential(dict(
            token_emb = nn.Embedding(config.vocab_size, config.num_embd),
            position_emb = nn.Embedding(config.block_size, config.num_embd),
            drop = nn.Dropout(config.dropout),
            head = nn.Sequential([Block(config) for _ in range(config.num_layer)]), # TODO: check if this is correct
            layer_norm = LayerNormalization(config.num_embd, config.bias)
        ))
        self.layer_softmax = nn.Linear(config.num_embd, config.vocab_size, bias=False)
        # tie weight of decoder and embedding layer, share weights.
        # this is done for efficiency: https://paperswithcode.com/method/weight-tying
        self.transformer.token_emb.weight = self.layer_softmax.weight

        # Apply the _init_weights function to initialize weights of all network layers. Provided in nn.Module
        self.apply(self._init_weights)
        # apply special scaled init to the residual projection
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                nn.init.normal_(p, mean=0, std=0.02/math.sqrt(2 * config.num_layer))
        
        print(f"number of parameters: {sum(p.numel() for p in self.parameters())}")
        
    def get_num_params(self, non_embedding=True):
        """
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.position_emb.numel()
        return n_params
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, f"Cannot forward, model block size is exhausted. Got {t} tokens, but configured for {self.config.block_size} context window"
        
        t_e = self.transformer.token_emb(idx)
        p_e = self.transformer.position_emb(torch.arange(t, device=device))
        x = self.transformer.drop(t_e + p_e)
        # TODO: check correct if Sequential is used
        x = self.transformer.head(x)
        x = self.transformer.layer_norm(x)
        
        if targets is not None:
            logits = self.layer_softmax(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),ignore_index=-1)
        else:
            # if no targets are provided, we return the next token prediction using the last token (ie: -1 on the T axis)
            logits = self.layer_softmax(x[:, [-1], :]) # using list [-1] to preserve the time dimension
            loss = None
            
        return logits, loss
                        
    @torch.no_grad()
    def generate(self, idx, max_len, temperature=1.0, top_k=None):
        """
        Given conditioning prefix tokens idx of shape (B, T), generate the next token until max_len. The prediction
        is fed back into the model for the next token. The model is in eval mode during generation.
        """
        for _ in range(max_len):
            # if the conditioning prefix exceeds the block size, truncate it
            idx_condition = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # input the conditioning prefix and generate the next token, no target is provided
            logits, _ = self(idx_condition)
            # pluck the logits at the final step and apply temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop probabilities to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1))) 
                logits[logits < v[:, [-1]]] = float('-inf') # -inf make the probability 0 after softmax
            # apply softmax to convert logits to normalized probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution or take the most likely
            idx_next = torch.multinomial(probs, num_samples=1)
            # concatenate the new token to the conditioning prefix ie: the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
        