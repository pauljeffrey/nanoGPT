from torch.utils.data import Dataset
import numpy as np
import torch
import random
import math


class TextDataset(Dataset):
    def __init__(self, data_path, window=64, block_size=256,split_type="train",device="cuda",shape=None):
        #self.index = 0
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r', shape=shape) #os.path.join(data_dir, 'train.bin')
        self.window = window
        self.block_size = block_size
        self.arr_len = len(self.data)
        print("Arr length: ", self.arr_len)
        self.split = split_type
        self.len = 1 + math.floor((self.arr_len - (self.block_size + 1)) / self.window)
        self.device = device
        
    def __getitem__(self, index):
        return self.get_batch(index)
        
    def __len__(self):
        return self.len
    
    def get_batch(self, i):
        if self.split == 'train':
            x = torch.from_numpy((self.data[i*self.window: (i*self.window)+ self.block_size]).astype(np.int64))
            y = torch.from_numpy((self.data[(i*self.window)+1: (i* self.window)+1+ self.block_size]).astype(np.int64))
            print("train_sample:- ", "X shape: ", x.shape, 'Y shape: ', y.shape)

        else:
            #i = random.choice(random.sample(range(len(self.data) - self.block_size), random.choice(range(self.len))))
            x = torch.from_numpy((self.data[i:i+self.block_size]).astype(np.int64))
    
            y = torch.from_numpy((self.data[i+1:i+1+self.block_size]).astype(np.int64))
            print("test sample:- ", "X shape: ", x.shape, 'Y shape: ', y.shape)
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        #x, y = x.pin_memory().to(self.device, non_blocking=True), y.pin_memory().to(self.device, non_blocking=True)
        
        #x, y = x.to(self.device), y.to(self.device)
        
        return x, y
        
        


def get_loader(data_path,
               batch_size,
               window = 64,
               block_size= 256,
               split_type= 'train',
               device= 'cuda',
                shape = None,
               shuffle = False,
               ):
    
    dataset = TextDataset(data_path,
                        window=window,
                        block_size = block_size,
                        split_type= split_type,
                        device= device,
                        shape= shape
                        )
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle= shuffle,
                                              num_workers = 4
                                              )
    return data_loader
        
