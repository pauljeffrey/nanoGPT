# saves the openwebtext dataset to a binary file for training. following was helpful:
# https://github.com/HazyResearch/flash-attention/blob/main/training/src/datamodules/language_modeling_hf.py

import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset # huggingface datasets
import datasets

# number of workers in .map() call
# good number to use is ~order number of cpu cores // 2
num_proc = 8

# number of workers in load_dataset() call
# best number might be different from num_proc above as it also depends on NW speed.
# it is better than 1 usually though
num_proc_load_dataset = num_proc

if __name__ == '__main__':
    # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
    #dataset = load_dataset("openwebtext", num_proc=num_proc_load_dataset)
    # Don't use 0, 9, 28| 1,2,23,31,17,15,16, 18
    data_files = [
        "data/train-00032-of-00035-65723db2a29abae8.parquet","data/train-00033-of-00035-bcb2a36aebfb89f9.parquet",
                  "data/train-00034-of-00035-3244e25f0c60266d.parquet","data/train-00029-of-00035-4fda4ad62c4ffb34.parquet",
                  "data/train-00030-of-00035-7722c3ba07048ce8.parquet", "data/train-00027-of-00035-35cffad4db0bf6b9.parquet",
                  "data/train-00026-of-00035-b0969ddc6407c018.parquet","train-00020-of-00035-64d5581b0d8c4437.parquet","train-00003-of-00035-64d5581b0d8c4437.parquet",]
    
    # "data/train-00001-of-00035-0ffa1b2c1533e462.parquet","data/train-00002-of-00035-8d4d29f0bb986f30.parquet",
    #               "data/train-00023-of-00035-1751103bdc6eb74c.parquet", "data/train-00031-of-00035-e8233b95e5b92059.parquet",
    #               "data/train-00017-of-00035-e3e493e9d916d4f5.parquet", "data/train-00015-of-00035-b51782f9289bc156.parquet",
    #               "data/train-00016-of-00035-5114eb1a53695860.parquet","data/train-00018-of-00035-aff94553959b76bb.parquet",
    #               "data/train-00019-of-00035-4a50511881e93615.parquet",]
    # download_config = datasets.DownloadConfig(force_download=True)
    # #https://huggingface.co/datasets/vietgpt/the_pile_openwebtext2/blob/main/data/
    dataset = load_dataset("vietgpt/the_pile_openwebtext2", num_proc=num_proc_load_dataset, data_files= data_files,ignore_verifications=True)#, cache_dir= "/content/drive/MyDrive/nanoGPT/.cache/train_dataset") #verification_mode = None,cache_dir= "/content/drive/MyDrive/nanoGPT/.cache/train_dataset",  download_config = download_config
    # owt by default only contains the 'train' split, so create a test split
    print(dataset)
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset['val'] = split_dataset.pop('test') # rename the test split to val


    # we now want to tokenize the dataset. first define the encoding function (gpt2 bpe)
    enc = tiktoken.get_encoding("gpt2")
    def process(example):
        ids = enc.encode_ordinary(example['text']) # encode_ordinary ignores any special tokens
        ids.append(enc.eot_token) # add the end of text token, e.g. 50256 for gpt2 bpe
        # note: I think eot should be prepended not appended... hmm. it's called "eot" though...
        out = {'ids': ids, 'len': len(ids)}
        return out

    # tokenize the dataset
    tokenized = split_dataset.map(
        process,
        remove_columns=['title', 'text', 'reddit_scores'],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # concatenate all the ids in each dataset into one large file we can use for training
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        print("Array length:" ,arr_len)
        #filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
        filename = os.path.join(os.path.abspath('/kaggle/working/'), f'{split}.bin')
        dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
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
