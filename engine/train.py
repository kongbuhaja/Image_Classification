import torch
import torch.nn as nn
import tqdm
from .eval import process as eval_process
from .lr import Scheduler

def process(model, train_loader, eval_loader, args, device='cuda:0'):
    optimizer = torch.optim.SGD(model.model.parameters(), lr=args.init_lr, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 0.01 + 0.99*(1-(e+1)/epochs))
    scheduler_fn = Scheduler(ilr=args.init_lr, epochs=args.epochs, wepochs=args.wepochs)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: scheduler_fn(e)/args.init_lr)

    model.model.train()
    for e in range(args.epochs):
        train_tqdm = tqdm.tqdm(train_loader, total=len(train_loader), ncols=121, desc=f'Train epochs {e+1}/{args.epochs}', ascii=' =', colour='red')
        train_loss = 0.
        lr = optimizer.param_groups[0]['lr']
        for iter, (x_data, y_data) in enumerate(train_tqdm):
            optimizer.zero_grad()

            pred = model.model(x_data.to(device))
            loss = loss_fn(pred, y_data.to(device))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_tqdm.set_postfix_str(f'| lr: {lr:.5f}, total_loss: {train_loss/(iter+1):.5f}')
    
        if (e)%args.eval_term == 0:
            recall, val_loss = eval_process(model, eval_loader, loss_fn, device=device)
            model.add_log(e+1, recall, val_loss, lr)
            model.save()

        scheduler.step()

        if e - model.best_epoch > args.patience:
            break
            