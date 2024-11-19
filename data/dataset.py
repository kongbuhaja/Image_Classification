import torch
import os
from PIL import Image
from .common import *

class TinyImageNetDataset(torch.utils.data.Dataset):
    names = []
    name2idx = {}
    idx2name = {}
    def __init__(self, split='train', x_transform=None, y_transform=None, cache=True):
        self.root = 'dataset/tiny-imagenet-200'
        self.split = split
        self.x_transform = x_transform
        self.y_transform = y_transform
        self.imgs = []
        self.targets = []
        self.cache = cache

        if split == 'train':
            train_dir = f'{self.root}/train'
            TinyImageNetDataset.set(sorted(os.listdir(f'{train_dir}')))
            for name in self.names:
                for img in os.listdir(f'{train_dir}/{name}/images'):
                    image = f'{train_dir}/{name}/images/{img}'
                    self.imgs += [self.read_img(image)] if cache else [image]
                    self.targets += [TinyImageNetDataset.name2idx[name]]
        elif split == 'val':
            val_dir = f'{self.root}/val'
            val_ann_file = f'{val_dir}/val_annotations.txt'
            images, names = self.read_val_annotations(val_ann_file)
            TinyImageNetDataset.set(sorted(list(set(names))))
            for img, name in zip(images, names):
                image = f'{val_dir}/images/{img}'
                self.imgs += [self.read_img(image)] if cache else [image]
                self.targets += [TinyImageNetDataset.name2idx[name]]
    
    @staticmethod
    def set(names):
        if not TinyImageNetDataset.names:
            TinyImageNetDataset.names = sorted(names)
            TinyImageNetDataset.name2idx = {n:i for i, n in enumerate(TinyImageNetDataset.names)}
            TinyImageNetDataset.idx2name = {i:n for i, n in enumerate(TinyImageNetDataset.names)}

    def read_img(self, img_path):
        return Image.open(img_path).convert('RGB')
    
    def read_val_annotations(self, val_ann_file):
        with open(val_ann_file, 'r') as f:
            images = []
            names = []
            for line in f.readlines():
                parts = line.strip().split('\t')
                images += [parts[0]]
                names += [parts[1]]
        return images, names

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        label = self.targets[idx]
        img = self.imgs[idx] if self.cache else self.read_img(self.imgs[idx])
        
        if self.x_transform:
            img = self.x_transform(img.copy())
        if self.y_transform:
            label = self.y_transform(label)
        
        return img, label