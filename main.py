import engine, data, nn
import torch

def train_process(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, eval_loader, test_loader, nc = data.get_dataloader(args,
                                                                     split='all')

    model = nn.Model(args.model, nc)
    model.model = model.model.to(device)

    engine.train_process(model, train_loader, eval_loader, args, device=device)
    model.load()
    trpfl = engine.eval_process(model, test_loader, device=device)
    model.save(trpfl)
    engine.gradcam_process(model, test_loader)

def eval_process(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_loader, _ = data.get_dataloader(args.data, 
                                         split='test')
    
    model = nn.Model(path=args.path)
    model.model = model.model.to(device)

    trpfl = engine.eval_process(model, test_loader, device=device)
    model.save(trpfl)
    engine.gradcam_process(model, test_loader)

if __name__ == '__main__':
    args = engine.args
    if args.process == 'train':
        train_process(args)
    elif args.process == 'val':
        eval_process(args)