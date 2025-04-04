import torch
import torchvision.transforms as transforms
from .dataset import *
import os

def get_dataloader(args, split='all', cache=True):
    assert args.data in ['tiny-imagenet']
    assert split in ['train', 'val', 'test', 'all']
    
    check_dataset(args.data)

    path = '../datasets'
    if args.data == 'tiny-imagenet':
        path += '/tiny-imagenet-200'
    
    nc = len(os.listdir(f'{path}/train'))

    train_x_transform = transforms.Compose([
        transforms.Resize(args.imgsz),
        Random_Rotate(),
        Random_Vflip(),
        Random_Hflip(),
        Color_jitter(),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(), # 0-255, (H,W,C) -> 0-1, (C, H, W)
    ])
    train_y_transform = transforms.Compose([
        Soft_Label(0.05, nc)
    ])
    eval_x_transform = transforms.Compose([
        transforms.Resize(args.imgsz),
        Normalize(mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(), # 0-255, (H,W,C) -> 0-1, (C, H, W)
    ])
    eval_y_transform = transforms.Compose([

    ])
    
    loaders = []
    if split in ['train', 'all']:
        data = TinyImageNetDataset('train',
                                    x_transform=train_x_transform,
                                    y_transform=train_y_transform,
                                    cache=cache)
        loader = torch.utils.data.DataLoader(data, 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             drop_last=True, 
                                             num_workers=8, 
                                             pin_memory=True,
                                             prefetch_factor=1)
        loaders += [loader]
    if split in ['val', 'all']:
        data = TinyImageNetDataset('val', 
                                    x_transform=eval_x_transform,
                                    y_transform=eval_y_transform,
                                    cache=cache)
        loader = torch.utils.data.DataLoader(data, 
                                             batch_size=args.batch_size, 
                                             shuffle=False, 
                                             drop_last=True,
                                             num_workers=2, 
                                             pin_memory=True,
                                             prefetch_factor=1)
        loaders += [loader]

    if split in ['test', 'all']:
        data = TinyImageNetDataset('val', 
                                    x_transform=eval_x_transform,
                                    y_transform=eval_y_transform,
                                    cache=cache)
        loader = torch.utils.data.DataLoader(data, 
                                             batch_size=1, 
                                             shuffle=False, 
                                             num_workers=8, 
                                             pin_memory=True,
                                             prefetch_factor=1)
        loaders += [loader]

    print(f'Success to load {args.data} {split} dataset')
    return *loaders, nc
