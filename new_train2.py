import os
import time
import math
import pickle
from contextlib import nullcontext
from dataset import *

import numpy as np
from accelerate import Accelerator, DeepSpeedPlugin
from tqdm import tqdm


from model import GPTConfig, GPT
from transformers import GPTJConfig, GPTJForCausalLM, get_scheduler
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 256
eval_iters = 500
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
window = 64
epochs = 100

# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gptj-1b' # 'run' + str(time.time())

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 1 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256 #1024

# model
n_positions=2048
rotary_dim = 64
n_layer = 18 #16
n_head = 20
n_embd = 1280 #768
bos_token_id = 50256
eos_token_id = 50256
n_inner = None
activation_function = "gelu_new"
layer_norm_epsilon = 1e-5
use_cache=True
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
device= "cuda"
device_type = 'cuda' if 'cuda' in device else 'cpu' 
scheduler = "cosine"

# system
#device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


# Create accelerator
deepspeed_plugin = DeepSpeedPlugin(zero_stage=3, gradient_accumulation_steps=gradient_accumulation_steps, gradient_clipping=1.0)
accelerator= Accelerator(mixed_precision='fp16', deepspeed_plugin=deepspeed_plugin)
#AcceleratorState().deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = batch_size
accelerator.wait_for_everyone()
device = accelerator.device



data_dir = os.path.join('data', dataset)
# train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
# val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
 

# Dataloader
train_dataloader = get_loader("/kaggle/working/train.bin",  batch_size,
               window = window, block_size= block_size, split_type= 'train', device= accelerator.device, shuffle = True) #os.path.join(data_dir, 'train.bin')

val_dataloader = get_loader("/kaggle/working/val.bin",  batch_size, window = window, block_size= block_size, split_type= 'eval',
               device= accelerator.device, shape = (block_size * eval_iters,), shuffle = False) #os.path.join(data_dir, 'val.bin')


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, rotary_dim=rotary_dim,n_inner=n_inner,
                  bias=bias, vocab_size=None, dropout=dropout, activation_function=activation_function, layer_norm_epsilon=layer_norm_epsilon,
                  use_cache=use_cache, bos_token_id=bos_token_id, eos_token_id=eos_token_id) # start with model_args from command line


if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = 50304
    conf = GPTJConfig(**model_args)
    model = GPTJForCausalLM(conf)
    
    # Memory problems
    model.to(device)

    print(f"Model created successfully. Model has {count_parameters(model) / 1e6} million parameters...")
    # optimizer
    optimizer_cls = (
                torch.optim.AdamW 
            )
    optimizer = optimizer_cls(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

    lr_scheduler = get_scheduler(
                                name= scheduler , optimizer=optimizer, num_warmup_steps=warmup_iters,
                                num_training_steps= max_iters,
                            )
    
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        conf = GPTConfig(**model_args)
        model = GPTJForCausalLM(conf)
        #state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        # unwanted_prefix = '_orig_mod.'
        # for k,v in list(state_dict.items()):
        #     if k.startswith(unwanted_prefix):
        #         print(K)
        #         state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(checkpoint['model'])
        
        print(f"Model checkpoint loaded successfully. Model has {count_parameters(model)/1e6} million parameters...")
        
        # optimizer
        optimizer_cls = (
                    torch.optim.AdamW 
                )
        
        optimizer = optimizer_cls(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

        lr_scheduler = get_scheduler(
                            name= scheduler, optimizer=optimizer, num_warmup_steps= warmup_iters,
                            num_training_steps= max_iters,
                        )
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        
    else:
        print(f"Checkpoint path: '{ckpt_path}' does not exist...")

# crop down the model block size if desired, using model surgery
# if block_size < model.config.block_size:
#     model.crop_block_size(block_size)
#     model_args['block_size'] = block_size # so that the checkpoint will have the right value



if compile:
    print("compiling the model.... (takes a ~minuter)")
    unoptimized_model = model
    model = torch.compile(model)
    
    
# Prepare accelerator
model, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader, val_dataloader)

checkpoint= None #free up memory


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    losses = []
    for X, Y in val_dataloader:
        output = model(input_ids=X, labels=Y)            
        losses.append(output.loss.item())
    out["eval"] = torch.mean(torch.tensor[losses])
    model.train()
    return out



#logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


t0 = time.time()


running_mfu = -1.0

progress_bar = tqdm(range(max_iters))

for epoch in range(epochs):
    train_losses = 0
    for iter_num , (X , Y) in enumerate(train_dataloader):
        train_loss = 0
        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            eval_loss = estimate_loss()
            print(f"step {iter_num}: train loss {train_losses/iter_num:.4f}, val loss {eval_loss['val']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": eval_loss['train'],
                    "val/loss": eval_loss['val'],
                    "lr": optimizer.lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
                
            if eval_loss['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = eval_loss['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                    
        if iter_num == 0 and eval_only:
            break

        
    
        outputs = model(input_ids=X, labels=Y)
        train_loss = outputs.loss.item() 
        loss = outputs.loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        accelerator.backward(loss)
            
        if iter_num % gradient_accumulation_steps == 0:
            # clip the gradient
            optimizer.step()
            lr_scheduler.step()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad()
            progress_bar.update(1)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % log_interval == 0:
            print(f"iter {iter_num}: loss {train_loss:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            
        train_losses += train_loss

        # termination conditions
        if iter_num > max_iters:
            break






