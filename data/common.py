import os
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageDraw

def check_dataset(data_name):
    if not os.path.exists('dataset'):
        os.mkdir('data')

    assert data_name in ['tiny-imagenet']

    if data_name == 'tiny_imagenet':
        if not os.path.exists('dataset/tiny-imagenet-200'):
            os.system('cd dataset \n \
                    wget http://cs231n.stanford.edu/tiny-imagenet-200.zip \n \
                    unzip tiny-imagenet-200.zip \n \
                    cd ../ \n \
                    clear')

        else:
            print('dataset/tiny-imagenet-200 already exists')

# y
class Soft_Label():
    def __init__(self, p, nc):
        self.p = p/nc
        self.v = 1 - p + self.p
        self.nc = nc

    def __call__(self, label):
        soft_label = np.zeros([self.nc]) + self.p
        soft_label[label] = self.v
        return soft_label

# x
# statistic
class Normalize():
    def __init__(self, mean=0.5, std=0.5):
        self.mean = np.array(mean).astype(np.float32)
        self.std = np.array(std).astype(np.float32)

    def __call__(self, img):
        return (np.array(img).astype(np.float32)/255. - self.mean) / self.std

# geometric
class Random_Rotate():
    def __init__(self, p=0.75):
        self.p = p

    def __call__(self, img):
        return img.rotate(random.choice([0, 90, 180, 270]))
    
class Random_Vflip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img
    
class Random_Hflip():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img
    
# pixel
class Color_jitter():
    def __init__(self, brightness=0.2, contrast=0.2, color=0.2):
        self.funcs = []
        self.values = []
        
        for func, v in zip(['Brightness', 'Contrast', 'Color'], [brightness, contrast, color]):
            if v:
                self.funcs += [getattr(ImageEnhance, func)]
                self.values += [v]

    def __call__(self, img):
        for func, v in zip(self.funcs, self.values):
            factor = random.uniform(max(0, 1-v), 1+v)
            img = func(img).enhance(factor)
        return img
    
class Random_erase():
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < 0.5:
            W, H = img.size
            x, y = random.randint(0, W), random.randint(0, H)
            EW = random.randint(int(W * 0.02), int(W * 0.33))
            EH = random.randint(int(H * 0.02), int(H * 0.33))
            color = tuple(random.randint(0, 255) for _ in range(3))
            draw = ImageDraw.Draw(img)
            draw.rectangle([x, y, x + EW, y + EH], fill=color)
        return img
    
