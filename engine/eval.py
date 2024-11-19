import torch
import torch.nn as nn
import tqdm

def process(model, eval_loader, loss_fn=None, device='cuda:0'):
    loss_fn = nn.CrossEntropyLoss() if loss_fn is None else loss_fn
    r, l = 0., 0.
    model.model.eval()
    with torch.no_grad():
        positive = 0
        val_loss = 0.
        val_tqdm = tqdm.tqdm(eval_loader, total=len(eval_loader), ncols=121, desc=f'Eval model', ascii=' =', colour='blue')
        for iter, (x_data, y_data) in enumerate(val_tqdm):
            pred = model.model(x_data.to(device))
            loss = loss_fn(pred, y_data.to(device))

            val_loss += loss.item()

            pred_ = torch.argmax(pred, dim=1).to('cpu')
            positive += torch.sum(pred_ == y_data).item()
            r = positive/(eval_loader.batch_size * (iter+1))
            l = val_loss/(iter+1)
            val_tqdm.set_postfix_str(f'| recall: {r:.5f}, val_loss: {l:.5f}')

    return r, l