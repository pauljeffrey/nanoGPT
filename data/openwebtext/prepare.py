# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
import argparse
from datasets import load_dataset # huggingface datasets
import datasets
from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PErsonalized Prompt Learning for Explainable Recommendation (PEPLER)')
    parser.add_argument('--data_path', type=str, default=None,
                        help='path to store the processed data downloaded from huggingface')
    parser.add_argument('--num_proc', type=int, default=8,
                        help='number of processes to use to prepare dataset')
    parser.add_argument('--dataset', type=str, default="vietgpt/the_pile_openwebtext2",
                        help='specify the dataset you want to use from huggingface datasets')
    parser.add_argument('--repo_name', type=str, default="gpt-j",
                        help='specify the hugging face repository to push tokenizer.')
     
    args = parser.parse_args()

    num_proc_load_dataset = args.num_proc
    
    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)
    
    dataset = load_dataset(args.dataset, num_proc=num_proc_load_dataset,verification_mode="no_checks", data_files= ["data/train-00032-of-00035-65723db2a29abae8.parquet"])#,
                                                                                                                    # "data/train-00033-of-00035-bcb2a36aebfb89f9.parquet",
                                                                                                                    # "data/train-00034-of-00035-3244e25f0c60266d.parquet","data/train-00029-of-00035-4fda4ad62c4ffb34.parquet",
                                                                                                                    # "data/train-00030-of-00035-7722c3ba07048ce8.parquet"])
    #, data_files= data_files,ignore_verifications=True)#, cache_dir= "/content/drive/MyDrive/nanoGPT/.cache/train_dataset") #verification_mode = None,cache_dir= "/content/drive/MyDrive/nanoGPT/.cache/train_dataset",  download_config = download_config
    # owt by default only contains the 'train' split, so create a test split
    print(dataset)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val


    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    #enc = tiktoken.get_encoding("gpt2")
    enc = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b", use_fast=True)
    enc.push_to_hub(args.repo_name, private=True)
    
    if enc.pad_token is None:
        enc.pad_token = enc.eos_token
        
    def process(example):
        #ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        result = enc(example['text'] + enc.eos_token)
        result["input_ids"]
        #ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': result["input_ids"], 'len': len(result["input_ids"])}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['title', 'text', 'reddit_scores'],
        desc="tokenizing the splits",
        num_proc=args.num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        #print("Array length:" ,arr_len)
        #filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
        
        if args.data_path is None:
            filename = os.path.join(os.path.dirname(__file__), f'{split}.bin') # Use current working directory as storage location.
        else:
            filename = os.path.join(args.data_path, f'{split}.bin')
            
        arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
        
        total_batches = 1024 if split == 'train' else 128

        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            # Batch together samples for faster write
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            # Write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()
        


    # train.bin is ~17GB, val.bin ~8.5MB
    # train has ~9B tokens (9,035,582,198)
    # val has ~4M tokens (4,434,897)

    # to read the bin files later, e.g. with numpy:
    # m = np.memmap('train.bin', dtype=np.uint16, mode='r')
