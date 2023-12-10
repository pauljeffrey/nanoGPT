import os
import time
import math
import pickle
from contextlib import nullcontext
from dataset import *

import numpy as np
from accelerate import Accelerator, DeepSpeedPlugin
from tqdm import tqdm
from bitsandbytes.optim import Adam8bit
from omegaconf import OmegaConf

from model import GPTConfig, GPT
from transformers import GPTJConfig, GPTJForCausalLM, get_scheduler, AdamW, Adafactor
import torch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O

config = OmegaConf.load("./config/gpt_j_config.yaml")


out_dir = config["out_dir"]
repo_name = config["repo_name"]
train_data_path = config["train_data_path"]
eval_data_path = config["eval_data_path"]
eval_interval = config["eval_interval"]
log_interval = config["log_interval"]
eval_iters = config["eval_iters"]
num_proc= config["num_proc"]
eval_only = config["eval_only"] # if True, script exits right after the first eval
push_to_hub_every=config["push_to_hub_every"]
always_save_checkpoint = config["always_save_checkpoint"] # if True, always save a checkpoint after each eval
init_from = config["init_from"] # 'scratch' or 'resume' or 'gpt2*'
epochs = config["epochs"]

# wandb logging
wandb_log = config["wandb_log"] # disabled by default
wandb_project = config["wandb_project"]
wandb_run_name = config["wandb_run_name"] # 'run' + str(time.time())

# data
dataset = config["dataset"]
gradient_accumulation_steps = config["gradient_accumulation_steps"] #128 # used to simulate larger batch sizes
train_batch_size = config["train_batch_size"] # if gradient_accumulation_steps > 1, this is the micro-batch size
eval_batch_size = config["eval_batch_size"]
block_size = config["block_size"] 


# model
rope_theta=config["rope_theta"]
n_layer = config["n_layer"] #28
n_head = config["n_head"]
n_embd = config["n_embd"]
bos_token_id = config["bos_token_id"]
eos_token_id = config["eos_token_id"]
pad_token_id = config["pad_token_id"]
use_cache = config["use_cache"]
activation_function = config["activation_functioin"]
layer_norm_epsilon = config["layer_norm_epsilon"] #1e-6
dropout = config["dropout"] # for pretraining 0 is good, for finetuning try 0.1+
vocab_size= config["vocab_size"]

# adamw optimizer
learning_rate = config["learning_rate"] #6e-4 # max learning rate
max_iters = config["max_iters"] #2000000 # total number of training iterations
weight_decay = config["weight_decay"] #1e-1
beta1 = config["beta1"] #0.9
beta2 = config["beta2"] #0.95
# learning rate decay settings
decay_lr = config["decay_lr"] #True # whether to decay the learning rate
warmup_iters = config["warmup_iters"] #2000 # how many steps to warm up for
lr_decay_iters = config["lr_decay_iters"] #600000 # should be ~= max_iters per Chinchilla
min_lr = config["min_lr"] #6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
device= config["device"] #"cuda"
device_type = 'cuda' if 'cuda' in device else 'cpu' 
scheduler = config["scheduler"] #"cosine"
zero_stage= config["zero_stage"] #2
gradient_clipping = config["gradient_clipping"] #1.0
optimizer_name = config["optimizer_name"]#"Adam8bit"


# model
rotary_dim = config["rotary_dim"]
n_inner = config["n_inner"]
bias = config["bias"] # do we use bias inside LayerNorm and Linear layers?


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
    #print("defaulting to vocab_size of 50400 (50257 rounded up for efficiency)")

    conf = GPTJConfig(**model_args)
    model = GPTJForCausalLM(conf)
    
    # Memory problems
    model.to(device)

    print(f"Model created successfully. Model has {count_parameters(model) / 1e6} million parameters...")
    # optimizer
    if optimizer_name == 'Adam8bit':
        optimizer = Adam8bit(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
        
    elif optimizer_name == 'Adafactor':
        
        optimizer = Adafactor(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)

    lr_scheduler = get_scheduler(
                                name= scheduler , optimizer=optimizer, num_warmup_steps=warmup_iters,
                                num_training_steps= max_iters,
                            )

    last_step = 0
    
elif init_from == 'local':
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
    
        model.load_state_dict(checkpoint['model'])
        model.to(device)
        print(f"Model checkpoint loaded successfully. Model has {count_parameters(model)/1e6} million parameters...")

        # optimizer
        if optimizer_name == 'Adam8bit':
            optimizer = Adam8bit(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
            
        elif optimizer_name == 'Adafactor':
            
            optimizer = Adafactor(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        else:
            optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
        
        lr_scheduler = get_scheduler(
                            name= scheduler, optimizer=optimizer, num_warmup_steps= warmup_iters,
                            num_training_steps= max_iters,
                        )
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        

        last_step = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    
elif init_from == 'hub':
    # init a new model from scratch
    print("Downloading model from hub ...")
    
    model = GPTJForCausalLM.from_pretrained(repo_name).to(device)
    

    print(f"Model loaded successfully. Model has {count_parameters(model) / 1e6} million parameters...")
    #Optimizer
    if optimizer_name == 'Adam8bit':
        optimizer = Adam8bit(model.parameters(), lr=learning_rate, betas=(beta1, beta2))
        
    elif optimizer_name == 'Adafactor':
        
        optimizer = Adafactor(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), weight_decay=weight_decay)
    

    lr_scheduler = get_scheduler(
                                name= scheduler , optimizer=optimizer, num_warmup_steps=warmup_iters,
                                num_training_steps= max_iters,
                            )

    last_step = 0


#Prepare accelerator
model, optimizer, lr_scheduler, train_dataloader, val_dataloader = accelerator.prepare(model, optimizer, lr_scheduler, train_dataloader, val_dataloader)

if compile:
    print("compiling the model.... (takes a ~minuter)")
    unoptimized_model = model
    model = torch.compile(model)
    

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
                
            # if eval_loss['eval'] < best_val_loss or always_save_checkpoint:
            #     best_val_loss = eval_loss['eval']
            # Create output directory if it doesn't exist
            # if not os.path.exists(out_dir):
            #     os.mkdir(out_dir)
            #     if iter_num > 0:
            #         checkpoint = {
            #             'model': accelerator.unwrap_model(model).state_dict(),
            #             'optimizer': optimizer.state_dict(),
            #             'lr_scheduler': lr_scheduler.state_dict(),
            #             'model_args': model_args,
            #             'iter_num': iter_num,
            #             'best_val_loss': best_val_loss,
            #             'config': config,
            #         }
                    
            #         print(f"saving checkpoint to {out_dir}")
                    
            #         torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
                    
        if iter_num % (push_to_hub_every * gradient_accumulation_steps) == 0 and iter_num != 0:
            accelerator.print(f"Pushing to HF hub...")
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            try:
                if accelerator.is_main_process:
                    unwrapped_model.push_to_hub(repo_name)

            except Exception as e:
                accelerator.print(e)
                accelerator.print(f"Failed to push to hub")
                    
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

    accelerator.print(f"Epoch {epoch} finished.")
    accelerator.print(f"Pushing to HF hub...")
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    try:
        if accelerator.is_main_process:
            unwrapped_model.push_to_hub(repo_name, private=True)

    except Exception as e:
        accelerator.print(e)
        accelerator.print(f"Failed to push to hub")





