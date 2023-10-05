import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
from accelerate import Accelerator
from accelerate import Accelerator, DistributedType ,DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import DummyOptim, DummyScheduler, set_seed

from model import GPTConfig, GPT
from transformers import GPTJConfig, GPTJForCausalLM
import torch


# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 256
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 4 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256 #1024

# model
n_positions=2048
rotary_dim = 64
bos_token_id = 50256
eos_token_id= 50256
n_layer = 32
n_head = 24 #16
n_embd = 1536 #2048
n_inner = None
activation_function = "gelu_new"
layer_norm_epsilon = 1e-5
use_cache=True
dropout = 0.05 # for pretraining 0 is good, for finetuning try 0.1+
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

# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

deepspeed_plugin = DeepSpeedPlugin(stage=3, gradient_accumulation_steps=gradient_accumulation_steps, gradient_clipping=1.0)
accelerator= Accelerator(mixed_precision='fp16', deepspeed_plugin=deepspeed_plugin)




train_data = np.memmap("/kaggle/working/nanoGPT/data/openwebtext/train.bin", dtype=np.uint16, mode='r') #os.path.join(data_dir, 'train.bin')
val_data = np.memmap("/kaggle/working/nanoGPT/data/openwebtext/val.bin", dtype=np.uint16, mode='r') #os.path.join(data_dir, 'val.bin')

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
# meta_path = os.path.join(data_dir, 'meta.pkl')
# meta_vocab_size = None
# if os.path.exists(meta_path):
#     with open(meta_path, 'rb') as f:
#         meta = pickle.load(f)
#     meta_vocab_size = meta['vocab_size']
#     print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")


# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, rotary_dim=64,n_inner=n_inner,
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
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

# Memory problems
model.to(device)

# optimizer
optimizer_cls = (
            torch.optim.AdamW 
        )
optimizer = optimizer_cls(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
    
checkpoint=None
    
    



