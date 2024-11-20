import engine, data, nn
import torch

def train_process(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, eval_loader, test_loader, nc = data.get_dataloader(args.data,
                                                                     split='all', 
                                                                     imgsz=args.imgsz,
                                                                     batch_size=args.batch_size)

    model = nn.Model(args.model, nc)
    model.model = model.model.to(device)

    engine.train_process(model, train_loader, eval_loader, args, device=device)
    model.load()
    rpfl = engine.eval_process(model, test_loader, device=device)
    engine.gradcam_process(model, test_loader)
    model.save(rpfl)

def eval_process(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_loader, _ = data.get_dataloader(args.data, 
                                         split='test',
                                         imgsz=args.imgsz,
                                         batch_size=1)
    model = nn.Model(path=args.path)
    model.model = model.model.to(device)

    engine.eval_process(model, test_loader, device=device)
    engine.gradcam_process(model, test_loader)

if __name__ == '__main__':
    args = engine.args
    if args.process == 'train':
        train_process(args)
    elif args.process == 'val':
        eval_process(args)