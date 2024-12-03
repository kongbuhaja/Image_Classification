import torch
import torch.nn as nn
import os
from .models import *
from data.common import UnNormalize
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class Model():
    def __init__(self, model_name='', nc=200, path=None):
        self.info = {'model_name':model_name,
                     'params':-1}
        self.best = {'epoch':-1,
                     'recall':-1,
                     'precision':-1,
                     'f1_score':-1,
                     'train_loss':-1,
                     'val_loss':-1}
        
        if path:
            self.info['model_name'] = path.split('/')[-1].split('_')[0]
            self.path = path
            self.load()
        else:
            self.info['model_name'] = model_name
            self.get_savepath()
            self.model = get_model(model_name, nc)
            print(f'Success to initialize {model_name} model')
    
        self.nc = nc
        self.info['params'] = self.count_parameters(verbose=0)
        self.epochs, self.lrs, self.train_losses = [], [], []
        self.recalls, self.precisions, self.f1_scores, self.val_losses = [], [], [], []
        

    def get_savepath(self):
        path = 'trained_models'
        if not os.path.exists(path):
            os.mkdir(path)
        count = sum(
            d.startswith(f"{self.info['model_name']}_") 
            for _, dirs, _ in os.walk(path) for d in dirs
        )
        path += f"/{self.info['model_name']}_{count+1}"
        os.mkdir(path)
        self.path = path

    def load(self):
        assert os.path.exists(f'{self.path}/model.pt')

        self.model = torch.load(f'{self.path}/model.pt')
        with open(f'{self.path}/model.info', 'r') as f:
            text = f.read()
        
        for paragraph in text.split('\n\n')[:2]:
            for line in paragraph.split('\n'):
                if ':' in line:
                    k, v = line.split(':')
                    k, v = k.strip(' '), v.strip(' ')
                    if k in self.best.keys():
                        self.best[k] = int(v)-1 if k == 'epoch' else v

        print(f'Success to load model from {self.path}')

    def add_log(self, e, tl, lr, r, p, f, vl):
        self.epochs += [e]
        self.train_losses += [tl]
        self.lrs += [lr]
        self.recalls += [r]
        self.precisions += [p]
        self.f1_scores += [f]
        self.val_losses += [vl]

    def save(self, trpfl=None):
        if trpfl is not None:
            self.save_info(trpfl)
        
        else:
            if self.best['recall'] < self.recalls[-1]:
                self.best['epoch'] = self.epochs[-1]
                self.best['recall'] = self.recalls[-1]
                self.best['precision'] = self.precisions[-1]
                self.best['f1_score'] = self.f1_scores[-1]
                self.best['train_loss'] = self.train_losses[-1]
                self.best['val_loss'] = self.val_losses[-1]

                torch.save(self.model, f'{self.path}/model.pt')
                self.save_info()
                print(f'Success to save model in {self.path}')
            self.save_csv()

    def save_info(self, trpfl=None, term=16):
        info_path = f'{self.path}/model.info'
        with open(info_path, 'w') as f:
            text = f"{'model_name':<{term}}:{self.info['model_name']:>{term}}\n" +\
                   f"{'params':<{term}}:{self.info['params']:>{term}}\n\n" +\
                   f"{'Train'}\n" +\
                   f"{'epoch':<{term}}:{int(self.best['epoch'])+1:>{term}}\n" +\
                   f"{'recall':<{term}}:{float(self.best['recall']):>{term}.8f}\n" +\
                   f"{'precision':<{term}}:{float(self.best['precision']):>{term}.8f}\n" +\
                   f"{'f1_score':<{term}}:{float(self.best['f1_score']):>{term}.8f}\n" +\
                   f"{'train_loss':<{term}}:{float(self.best['train_loss']):>{term}.8f}\n" +\
                   f"{'val_loss':<{term}}:{float(self.best['val_loss']):>{term}.8f}\n\n"
            if trpfl is not None:
                text += f"{'Evaluation'}\n" +\
                        f"{'process_time':<{term}}:{trpfl[0]:>{term}.8f}\n" +\
                        f"{'recall':<{term}}:{trpfl[1]:>{term}.8f}\n" +\
                        f"{'precision':<{term}}:{trpfl[2]:>{term}.8f}\n" +\
                        f"{'f1_score':<{term}}:{trpfl[3]:>{term}.8f}\n" +\
                        f"{'loss':<{term}}:{trpfl[4]:>{term}.8f}"
                print(text)
            f.write(text)

    def save_csv(self, term=12):
        csv_path = f'{self.path}/result.csv'
        with open(csv_path, 'w') as fs:
            fs.write(f"{'epoch':<{term}}{'recall':<{term}}{'precision':<{term}}{'f1_score':<{term}}{'train_loss':<{term}}{'val_loss':<{term}}{'lr':<{term}}\n")
            for e, r, p, f, tl, vl, lr in zip(self.epochs, self.recalls, self.precisions, self.f1_scores, self.train_losses, self.val_losses, self.lrs):
                fs.write(f'{e+1:<{term}}{r:<{term}.6f}{p:<{term}.6f}{f:<{term}.6f}{tl:<{term}.6f}{vl:<{term}.6f}{lr:<{term}.6f}\n')

    def count_parameters(self, verbose=1):
        total_params = 0
        params = {}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                print('???')
                continue
            param_count = parameter.numel()
            total_params += param_count
            if verbose:
                if verbose==1:
                    block = name.split('.')[0]
                    if block not in params:
                        params[block] = param_count
                    else:
                        params[block] += param_count
                elif verbose==2:
                    params[name] = param_count
    
        print(f"Total trainable parameters: {total_params:,}")

        if verbose:
            print("Layer-wise parameter counts:")
            for name, count in params.items():
                print(f"{name}: {count:,}")
        
        return total_params
    
class GradCAM():
    def __init__(self, model):
        self.model = model.model
        self.get_savepath(model.path)
        if hasattr(self.model, 'psa'):
            self.target_layer = self.model.psa
        elif hasattr(self.model, 'psd'):
            self.target_layer = self.model.psd
        else:
            self.target_layer = self.model.layer4[-1]
        
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def visualize(self, img, heatmap, alpha=0.5):
        img = UnNormalize()(img)

        heatmap = Image.fromarray(heatmap).resize(img.shape[:2][::-1], Image.BICUBIC)
        heatmap = np.array(heatmap)

        colored_heatmap = plt.get_cmap('jet')(heatmap)[:, :, :3]

        imposed_img = (1-alpha) * img + alpha * (colored_heatmap)

        return (imposed_img * 255).astype(np.uint8)
    
    def get_savepath(self, path):
        self.path = f'{path}/images'

        if not os.path.exists(self.path):
            os.mkdir(self.path)

        self.count = sum(
            f.startswith(f'heatmap_pred') 
            for _, _, files in os.walk(self.path) for f in files
        )

    def save_heatmap(self, heatmaps):
        for label, heatmap in zip(['label', 'pred'], heatmaps):
            image = Image.fromarray(heatmap)
            image.save(f'{self.path}/heatmap_{label}_{self.count+1}.png')
        self.count += 1
    


def get_model(model_name, nc, c=32):
    model_name = model_name.lower()
    assert model_name in ['resnet18', 
                          'resnet182', 
                          'dresnet18', 
                          'dresnet182', 
                          'psaresnet18', 
                          'psdresnet18',
                          'psadresnet18', 
                          'psddresnet18',
                          'c2psaresnet18',
                          'c2psdresnet18',
                          'c2psadresnet18',
                          'c2psddresnet18', 
                          'resnet50',
                          'psdresnet182',
                          'psddresnet182',
                          'dresnet182',
                          'psadresnet182',]

    if model_name == 'resnet18':
        model = ResNet18(nc, c)
    elif model_name == 'dresnet18':
        model = DResNet18(nc, c)
    elif model_name == 'psaresnet18':
        model = PSAResNet18(nc, c)
    elif model_name == 'psdresnet18':
        model = PSDResNet18(nc, c)
    elif model_name == 'psadresnet18':
        model = PSADResNet18(nc, c)
    elif model_name == 'psddresnet18':
        model = PSDDResNet18(nc, c)
    elif model_name == 'c2psaresnet18':
        model = C2PSAResNet18(nc, c)
    elif model_name == 'c2psdresnet18':
        model = C2PSDResNet18(nc, c)
    elif model_name == 'c2psadresnet18':
        model = C2PSADResNet18(nc, c)
    elif model_name == 'c2psddresnet18':
        model = C2PSDDResNet18(nc, c)
    elif model_name == 'resnet50':
        model = ResNet50(nc, c)

    elif model_name == 'psddresnet182':
        model = PSDDResNet182(nc, c)
    elif model_name == 'psdresnet182':
        model = PSDResNet182(nc, c)
    elif model_name == 'dresnet182':
        model = DResNet182(nc, c)
    elif model_name == 'psadresnet182':
        model = PSADResNet182(nc, c)

    return model