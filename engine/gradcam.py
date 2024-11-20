import torch
import torch.nn as nn
import tqdm
from nn.model import GradCAM
from itertools import islice

def process(model, eval_loader, length=4, device='cuda:0'):
    gradcam = GradCAM(model)

    gradcam.model.eval()
    length = min(len(eval_loader), length)
    limited_loader = islice(eval_loader, length)

    val_tqdm = tqdm.tqdm(limited_loader, total=length, ncols=121, desc=f'Eval model', ascii=' =', colour='green')
    for iter, (x_data, y_data) in enumerate(val_tqdm):
        pred = gradcam.model(x_data.to(device))
        ci = pred.argmax(dim=1)

        heatmaps = []
        for idx in [y_data, ci]:
            one_hot = torch.zeros_like(pred)
            one_hot[0][idx] = 1
            gradcam.model.zero_grad()

            pred.backward(gradient=one_hot, retain_graph=True)

            pooled_gradients = torch.mean(gradcam.gradients, dim=[0, 2, 3])
            for i in range(gradcam.activations.shape[1]):
                gradcam.activations[:, i, :, :] *= pooled_gradients[i]

            heatmap = torch.mean(gradcam.activations, dim=1).squeeze()
            heatmap = torch.nn.functional.relu(heatmap)
            heatmap /= torch.max(heatmap)

            imposed_img = gradcam.visualize(x_data[0].permute(1,2,0).numpy(), heatmap.detach().cpu().numpy())
            heatmaps += [imposed_img]

        gradcam.save_heatmap(heatmaps)

