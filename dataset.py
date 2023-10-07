from torch.utils.data import Dataset
import numpy as np



class TextDataset(Dataset):
    def __init__(self, train_data, val_data, window=64, block_size=256, batch_size=16, arr_len=1e9):
        self.index = 0
        train_data = np.memmap(train_data, dtype=np.uint16, mode='r') #os.path.join(data_dir, 'train.bin')
        val_data = np.memmap(val_data, dtype=np.uint16, mode='r') #os.path.join(data_dir, 'val.bin')
        self.batch_size = batch_size
        self.window = window
        self.block_size = block_size
        self.arr_len = 
        
    def __getitem__(self, index) -> Any:
        return 
        
    def __len__(self):
        return self.arr_len
        
        


def get_loader(image_dir,
               caption_json,
               history_json,
               file_list,
               vocabulary,
               vocabulary2,
               transform,
               batch_size,
               s_max=10,
               n_max=50,
               shuffle=False,
               collate_fn=collate_fn
               ):
    dataset = (image_dir=image_dir,
                               caption_json=caption_json,
                               history_json = history_json,
                               file_list=file_list,
                               vocabulary=vocabulary,
                               vocabulary2 = vocabulary2,
                               s_max=s_max,
                               n_max=n_max,
                               transforms=transform)
    
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              collate_fn=collate_fn)
    return data_loader
        
batch_slice = window * (batch_size - 1) + block_size 

# Dataloader
def get_batch(split, mode = "eval"):
    if mode == 'train' and split == 'train':
        data = train_data
        global index
        ix = range( index, batch_slice, window)
        x = torch.stack([torch.from_numpy((data[i: i+ block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1: i+1+block_size]).astype(np.int64)) for i in ix])
        index += batch_slice
    else:
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