import torch
import torch.nn as nn
import torchvision
import os
from .models import *
from data.common import UnNormalize
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class Model():
    def __init__(self, model_name='', nc=200, path=None):
        if path:
            self.model_name = path.split('/')[-2]
            self.path = path
            self.load()
        else:
            self.model_name = model_name
            self.get_savepath()
            self.model = get_model(model_name, nc)
            print(f'Success to initialize {self.model_name} model')
    
        self.params = self.count_parameters(verbose=0)
        self.epochs, self.recalls, self.losses, self.lrs = [], [], [], []
        self.best_recall, self.best_epoch = -1, -1

    def get_savepath(self):
        path = 'trained_models'
        if not os.path.exists(path):
            os.mkdir(path)
        count = sum(
            d.startswith(f'{self.model_name}_') 
            for _, dirs, _ in os.walk(path) for d in dirs
        )
        path += f'/{self.model_name}_{count+1}'
        os.mkdir(path)
        self.path = path

    def load(self):
        assert os.path.exists(f'{self.path}/model.pt')

        self.model = torch.load(f'{self.path}/model.pt')
        with open(f'{self.path}/model.info', 'r') as f:
            text = f.read()
        self.best_recall = float(text.split('recall:')[1].split('\n')[0])
        self.best_epoch = int(text.split('epoch:')[1].split('\n')[0])
        self.params = int(text.split('params:')[1].split('\n')[0])
        if 'test_recall' in text:
            self.test_recall = float(text.split('test_recall:')[1].split('\n')[0])
            self.test_loss = float(text.split('test_loss:')[1].split('\n')[0])
        print(f'Success to load model from {self.path}')
        print(text)

    def add_log(self, epoch, recall, loss, lr):
        self.epochs += [epoch]
        self.recalls += [recall]
        self.losses += [loss]
        self.lrs += [lr]

    def save(self, test_recall=None, test_loss=None):
        if test_recall is not None and test_loss is not None:
            self.save_info(test_recall, test_loss)
        
        else:
            if self.best_recall < self.recalls[-1]:
                self.best_recall = self.recalls[-1]
                self.best_epoch = self.epochs[-1]

                torch.save(self.model, f'{self.path}/model.pt')
                self.save_info()
                print(f'Success to save model in {self.path}')
            self.save_csv()

    def save_info(self, test_recall=None, test_loss=None):
        info_path = f'{self.path}/model.info'
        with open(info_path, 'w') as f:
            text = f'model_name:{self.model_name}\n' +\
                   f'params:{self.params}\n' +\
                   f'epoch:{self.best_epoch+1}\n' +\
                   f'recall:{self.best_recall}\n'
            if test_recall is not None and test_loss is not None:
                text += f'test_recall:{test_recall}\n' +\
                        f'test_loss:{test_loss}\n'
            f.write(text)

    def save_csv(self, term=12):
        csv_path = f'{self.path}/result.csv'
        with open(csv_path, 'w') as f:
            f.write(f"{'epoch':<{term}}{'recall':<{term}}{'val_loss':<{term}}{'lr':<{term}}\n")
            for e, r, l, lr in zip(self.epochs, self.recalls, self.losses, self.lrs):
                f.write(f'{e+1:<{term}}{r:<{term}.5f}{l:<{term}.5f}{lr:<{term}.5f}\n')

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
                          'psddresnet18',
                          'c2psaresnet18',
                          'c2psdresnet18',
                          'c2psddresnet18', 
                          'resnet50', 
                          'torch_resnet18']

    if model_name == 'resnet18':
        model = ResNet18(nc, c)
    elif model_name == 'resnet182':
        model = ResNet182(nc, c)
    elif model_name == 'dresnet18':
        model = DResNet18(nc, c)
    elif model_name == 'dresnet182':
        model = DResNet182(nc, c)
    elif model_name == 'psaresnet18':
        model = PSAResNet18(nc, c)
    elif model_name == 'psdresnet18':
        model = PSDResNet18(nc, c)
    elif model_name == 'psddresnet18':
        model = PSDDResNet18(nc, c)
    elif model_name == 'c2psaresnet18':
        model = C2PSAResNet18(nc, c)
    elif model_name == 'c2psdresnet18':
        model = C2PSDResNet18(nc, c)
    elif model_name == 'c2psddresnet18':
        model = C2PSDDResNet18(nc, c)
    elif model_name == 'resnet50':
        model = ResNet50(nc, c)
    elif model_name == 'torch_resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(3, c, 3, 1)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(c*2**3, nc)
    
    return model