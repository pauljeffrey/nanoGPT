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

out_dir = './out'
repo_name = "gpt-j"
train_data_path = "./train.bin"
eval_data_path = "./val.bin"
eval_interval = 2048
log_interval = 512
eval_iters = 500
num_proc= 8
eval_only = False # if True, script exits right after the first eval
push_to_hub_every=32
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
gradient_accumulation_steps = 32 #128 # used to simulate larger batch sizes
train_batch_size = 2 # if gradient_accumulation_steps > 1, this is the micro-batch size
eval_batch_size = train_batch_size * 2
block_size = 256 #1024

# model
n_positions=2048
rotary_dim = 64
n_layer = 24 #28
n_head = 24 
n_embd = 1536 
bos_token_id = 50256
eos_token_id = 50256
n_inner = None
activation_function = "gelu_new"
layer_norm_epsilon = 1e-5
use_cache=True
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
vocab_size=50400

# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 2000000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
device= "cuda"
device_type = 'cuda' if 'cuda' in device else 'cpu' 
scheduler = "cosine"
zero_stage=3
gradient_clipping = 1.0

# system
#device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------


#print("Train_path: ",train_data_path)

# Create accelerator
deepspeed_plugin = DeepSpeedPlugin(zero_stage=zero_stage, gradient_accumulation_steps=gradient_accumulation_steps, gradient_clipping=gradient_clipping)
accelerator= Accelerator(mixed_precision='fp16', deepspeed_plugin=deepspeed_plugin)
accelerator.wait_for_everyone()
device = accelerator.device


# Create output directory if it doesn't exist
if not os.path.exists(out_dir):
    os.mkdir(out_dir)


print(train_batch_size, eval_batch_size)
# Dataloader
train_dataloader = get_loader(train_data_path,  train_batch_size,
               window = window, block_size= block_size, split_type= 'train', device= accelerator.device, shuffle = True) 

val_dataloader = get_loader(eval_data_path,  eval_batch_size, window = window, block_size= block_size, split_type= 'eval',
               device= accelerator.device, shape = (block_size * eval_iters,), shuffle = False) 


# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, rotary_dim=rotary_dim,n_inner=n_inner,
                  bias=bias, vocab_size=vocab_size, dropout=dropout, activation_function=activation_function, layer_norm_epsilon=layer_norm_epsilon,
                  use_cache=use_cache, bos_token_id=bos_token_id, eos_token_id=eos_token_id) # start with model_args from command line


if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    print("defaulting to vocab_size of 50400 (50257 rounded up for efficiency)")

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
    #Prepare accelerator
    model, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader, val_dataloader)

    last_step = 0
    
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
        conf = GPTJConfig(**model_args)
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
        model.to(device)
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
        
        #Prepare accelerator
        model, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader, val_dataloader)

        last_step = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        
    else:
        print(f"Checkpoint path: '{ckpt_path}' does not exist...")
        last_step = 0
# crop down the model block size if desired, using model surgery
# if block_size < model.config.block_size:
#     model.crop_block_size(block_size)
#     model_args['block_size'] = block_size # so that the checkpoint will have the right value



if compile:
    print("compiling the model.... (takes a ~minuter)")
    unoptimized_model = model
    model = torch.compile(model)
    
    
# Prepare accelerator
#model, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader, val_dataloader)

checkpoint= None #free up memory


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    losses = []
    for X, Y in val_dataloader:
        output = model(input_ids=X, labels=Y)            
        losses.append(output.loss.item())
    out["eval"] = torch.mean(torch.tensor(losses))
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
        if iter_num < last_step:
            continue
        train_loss = 0
        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0:
            eval_loss = estimate_loss()
            if train_losses > 0:
                print(f"step {iter_num}: train loss {train_losses/(iter_num+1):.4f}, val loss {eval_loss['eval']:.4f}")
            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": train_losses/(iter_num+1),
                    "val/loss": eval_loss['eval'],
                    "lr": optimizer.lr,
                    "mfu": running_mfu*100, # convert to percentage
                })
                
            if eval_loss['eval'] < best_val_loss or always_save_checkpoint:
                best_val_loss = eval_loss['eval']
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
                    
            # if iter_num % (push_to_hub_every * gradient_accumulation_steps) == 0 and iter_num != 0:
            #     accelerator.print(f"Pushing to HF hub...")
            #     accelerator.wait_for_everyone()
            #     unwrapped_model = accelerator.unwrap_model(model)
            #     try:
            #         if accelerator.is_main_process:
            #             unwrapped_model.push_to_hub(repo_name)

            #     except Exception as e:
            #         accelerator.print(e)
            #         accelerator.print(f"Failed to push to hub")
                    
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

    # accelerator.print(f"Epoch {epoch} finished.")
    # accelerator.print(f"Pushing to HF hub...")
    # accelerator.wait_for_everyone()
    # unwrapped_model = accelerator.unwrap_model(model)
    # try:
    #     if accelerator.is_main_process:
    #         unwrapped_model.push_to_hub(repo_name, private=True)

    # except Exception as e:
    #     accelerator.print(e)
    #     accelerator.print(f"Failed to push to hub")





