import torch
import torch.nn as nn
import tqdm
from .metric import Metric

def process(model, eval_loader, loss_fn=None, device='cuda:0'):
    loss_fn = nn.CrossEntropyLoss() if loss_fn is None else loss_fn
    model.model.eval()
    metric = Metric(model.nc)
    with torch.no_grad():
        positive = 0
        val_loss = 0.
        val_tqdm = tqdm.tqdm(eval_loader, total=len(eval_loader), ncols=161, desc=f'Eval model', ascii=' =', colour='blue')
        for iter, (x_data, y_data) in enumerate(val_tqdm):
            pred = model.model(x_data.to(device))
            loss = loss_fn(pred, y_data.to(device))

            val_loss += loss.item()

            pred_ = torch.argmax(pred, dim=1).to('cpu')
            positive += torch.sum(pred_ == y_data).item()
            metric.update(pred_.numpy(), y_data.numpy())
            r, p, f = metric.stat()
            l = val_loss/(iter+1)
            val_tqdm.set_postfix_str(f'| recall: {r:.5f}, precision: {p:.5f}, f1_score: {f:.5f}, val_loss: {l:.5f}')

    return r, p, f, l