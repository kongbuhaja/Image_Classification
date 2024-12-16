import torch
import torch.nn as nn
import tqdm
from .metric import Metric
import time

def process(model, eval_loader, loss_fn=None, device='cuda:0'):
    loss_fn = nn.CrossEntropyLoss() if loss_fn is None else loss_fn
    model.model.eval()
    metric = Metric(model.nc)
    with torch.no_grad():
        x_data = torch.randn(1, 3, 640, 640).cuda()

        for _ in range(10):
            _ = model.model(x_data)

        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        val_loss = 0.
        times = 0.
        val_tqdm = tqdm.tqdm(eval_loader, total=len(eval_loader), ncols=180, desc=f'Eval model', ascii=' =', colour='blue')
        for iter, (x_data, y_data) in enumerate(val_tqdm):
            x_input = x_data.to(device)
            # st = time.time()
            starter.record()
            pred = model.model(x_input)
            # times += (time.time() - st)
            ender.record()
            torch.cuda.synchronize()
            times += starter.elapsed_time(ender)
            loss = loss_fn(pred, y_data.to(device))
            val_loss += loss.item()

            pred_ = torch.argmax(pred, dim=1).to('cpu')
            metric.update(pred_.numpy(), y_data.numpy())
            r, p, f = metric.stat()
            l = val_loss/(iter+1)
            t = times/(iter+1)
            val_tqdm.set_postfix_str(f'| process time: {t:.5f} recall: {r:.5f}, precision: {p:.5f}, f1_score: {f:.5f}, val_loss: {l:.5f}')

    return t, r, p, f, l