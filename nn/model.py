import torch
import torch.nn as nn
import torchvision
import os
from .models import *

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

        self.model_name = model_name
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
                   f'epoch:{self.best_epoch}\n' +\
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
                f.write(f'{e:<{term}}{r:<{term}.5f}{l:<{term}.5f}{lr:<{term}.5f}\n')

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

def get_model(model_name, nc, c=64):
    model_name = model_name.lower()
    assert model_name in ['resnet18', 'dresnet18', 'resnet50', 'torch_resnet18']

    if model_name == 'resnet18':
        model = ResNet18(nc, c=c)

    elif model_name == 'dresnet18':
        model = DResNet18(nc, c=c)

    elif model_name == 'resnet50':
        model = ResNet50(nc, c=c)

    elif model_name == 'torch_resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        model.conv1 = nn.Conv2d(3, c, 3, 1)
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(c*2**3, nc)
    
    return model