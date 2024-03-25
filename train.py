"""
This training script can run on a single GPU (cuda or mps) and also on multiple
GPUs with distributed data parallel (ddp).

To run on single GPU/CPU:
$ python train.py 

To run on multiple GPUs on a single node/machine:
$ torchrun --standalone --nproc_per_node=gpu train.py

"""
from tqdm import tqdm
import torch
import time

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from contextlib import nullcontext

from src.model import GPT, GPTConfig

# -----------------------------------------------------------------
backend = 'nccl' # 'nccl', 'gloo', etc.
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'shakespeare-gpt2' # 'your_project_name_here
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# system
device_type = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
device = torch.device(device_type)
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
compile = False # use PyTorch 2.0 to compile the model to be faster
block_size = 1024
# data
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 64 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256
# model
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
num_layer = 6
num_head = 6
num_embd = 384
dropout = 0.2 # for pretraining 0 is good, for finetuning try 0.1+
bias = False
total_epochs = 5000
eval_iters = 200
save_every = 500
log_interval = -1
eval_interval = 500
init_from = 'scratch' # 'scratch' or 'resume' 
# model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                #   bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
# adamw optimizer
learning_rate = 3e-4 # max learning rate

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# -----------------------------------------------------------------

class Trainer:
    def __init__(
        self,
        model = torch.nn.Module,
        dataset = "dict[str, torch.long]",
        optimizer = torch.optim.Optimizer,
        snapshot_path = str,
    ):
        self.master_process = True
        self.gpu_id = 0
        self.device = device_type
        self.seed_offset = 0
        self.ddp_world_size = 1
        if ddp:
            self.ddp_rank = int(os.environ['RANK'])
            self.gpu_id = int(os.environ['LOCAL_RANK'])
            self.ddp_world_size = int(os.environ['WORLD_SIZE'])
            self.device = f'cuda:{self.gpu_id}'
            torch.cuda.set_device(self.device)
            self.master_process = self.ddp_rank == 0 # this process will do logging, checkpointing etc.
            self.seed_offset = self.ddp_rank # each process gets a different seed
        self.model = model.to(self.gpu_id)
        
        torch.manual_seed(1337 + self.seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
        
        self.tokens_per_iter = gradient_accumulation_steps * self.ddp_world_size * batch_size * block_size
        print(f"tokens per iteration will be: {self.tokens_per_iter:,}")
        
        if compile:
            print("compiling the model... (takes a ~minute)")
            self.unoptimized_model = self.model
            self.model = torch.compile(self.model) # requires PyTorch 2.0
        
        self.snapshot_path = snapshot_path
        if init_from == 'resume' and os.path.exists(self.snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()  
            
        if ddp:
            # after loading the snapshot, we will wrap the model with DDP
            self.model = DDP(self.model, device_ids=[self.gpu_id])
            
        self.dataset = dataset
        self.optimizer = optimizer
        self.epochs_run = 0
        self.best_val_loss = float('inf')
        
        # logging
        if wandb_log and self.master_process:
            import wandb
            wandb.init(project=wandb_project, name=wandb_run_name)
    
    def _save_snapshot(self, epoch):
        snapshot = {
            'model': self.model.state_dict() if not ddp else self.model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epochs_run': epoch,
            # 'model_args': model_args,
            'best_val_loss': self.best_val_loss,
            # 'config': config,
        }
        torch.save(snapshot, self.snapshot_path)
        
    def _load_snapshot(self):
        device_loc = f"cuda:{self.gpu_id}" if 'cuda' in device_type else device_type
        snapshot = torch.load(self.snapshot_path, map_location=device_loc)
        self.model.load_state_dict(snapshot["model"])
        self.epochs_run = snapshot["epochs_run"]
        print(f"resuming training from path {self.snapshot_path} at epoch: {self.epochs_run}")
        
    def sample_batch(self, split):
        data = self.dataset[split]
        ix = torch.randint(0, len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y
        
    def _run_batch(self, X, Y):
        
        
        self.optimizer.zero_grad()
        logits, loss = self.model(X, Y)
        loss.backward()
        self.optimizer.step()
        return loss.item()
            
    def _run_epoch(self, epoch, xb, yb):
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            with ctx:
                logits, loss = model(xb, yb)
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            xb, yb = self.sample_batch('train')
            if log_interval>0:
                print(f"[GPU:{self.gpu_id}] Epoch {epoch} | Input size {xb.shape} | Output size {yb.shape}")
                
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        
        return loss.item()
            
    # for checking the loss on the entire dataset, set model to eval and then to train before returning
    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = self.sample_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    def train(self, max_epochs: int):
        t0 = time.time()     
        xb, yb = self.sample_batch('train') # fetch the very first batch
        for epoch in tqdm(range(self.epochs_run, max_epochs), desc="Training Progress"):
            
            if epoch % save_every == 0 and self.master_process:       
                self._save_snapshot(epoch)
            
            # evaluate the loss on train/val sets and write checkpoints
            if epoch % eval_interval == 0 and self.master_process:
                losses = self.estimate_loss()
                print(f"epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
                if wandb_log:
                    import wandb
                    wandb.log({
                        "epoch": epoch,
                        "train/loss": losses['train'],
                        "val/loss": losses['val'],
                        "lr": learning_rate,
                    })
                if losses['val'] < self.best_val_loss:
                    best_val_loss = losses['val']
                    if epoch > 0:
                        self._save_snapshot(epoch)
                        print(f"training snapshot saved at {self.snapshot_path}")
               
            lossf = self._run_epoch(epoch, xb, yb)
            
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
    
            if log_interval>0 and epoch % log_interval == 0 and self.master_process:
                print(f"epoch {epoch}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
                
                
def MyTrainDataset(train_split):
    with open('data/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda x: [stoi[ch] for ch in x]
    decode = lambda x: ''.join([itos[i] for i in x])
    
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(train_split*len(data))
    train_data = data[:n]
    val_data = data[n:]
    return train_data, val_data if train_split<1 else None, decode, vocab_size
    
def load_train_objs():
    # dataset
    train_set, val_set, _, vocab_size = MyTrainDataset(train_split=0.9)  # load your dataset
    # model
    gptconf = GPTConfig(
            num_layer=num_layer, 
            num_head=num_head, 
            num_embd=num_embd, 
            block_size=block_size,
            bias=bias, 
            vocab_size=vocab_size,
            dropout=dropout)
    model = GPT(gptconf)
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    return {'train': train_set, 'val': val_set}, model, optimizer


if __name__ == "__main__":
    
    if ddp:
        init_process_group(backend=backend)
    dataset, model, optimizer = load_train_objs()
    trainer = Trainer(model, dataset, optimizer, 'snapshot.pth')
    trainer.train(total_epochs)
    if ddp:
        destroy_process_group()
    
    