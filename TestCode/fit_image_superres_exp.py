# while keeping the latent dimension fixed, it is then possible to fit an image

import os

import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
from tqdm import tqdm, trange
import random


def seed_everything(seed=777):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class PriorModules(nn.Module):
    def __init__(self, latent_size, num_modules=4):
        super(PriorModules, self).__init__()
        self.module_list = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_size, 4),
                nn.ReLU(),
                nn.Linear(4, 3)
            ) for i in range(num_modules)
        ])

    def forward(self, grid, attention):
        outs = []
        for module in self.module_list:
            outs.append(torch.sigmoid(module(grid)))

        outs = torch.stack(outs, dim=-2)
        # plt.imshow(torch.sigmoid(outs[0].detach().cpu()))
        # plt.show()
        final_output = (attention[..., None] * outs).sum(dim=-2)

        return final_output


class Model(nn.Module):
    def __init__(self, H, W, latent_size=16, scale_factor=2, num_modules=16):
        super(Model, self).__init__()
        self.grid = torch.nn.Parameter(torch.zeros(1, latent_size, H // scale_factor, W // scale_factor))
        self.attention_net = nn.Sequential(
            nn.Linear(latent_size, num_modules),
            nn.Softmax(dim=-1)
        )
        self.prior_modules = PriorModules(latent_size=latent_size, num_modules=num_modules)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bicubic')

    def forward(self):
        grid = self.upsample(self.grid).squeeze(0).permute(1, 2, 0)
        attention = self.attention_net(grid)
        out = self.prior_modules.forward(grid, attention)

        return out


mse2psnr = lambda x: -10. * torch.log10(x)
if __name__ == "__main__":
    scale_factor = 2
    seed_everything()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    os.makedirs('outputs_fixed_l', exist_ok=True)
    image = np.array(Image.open('r_0.png'))[..., :3] / 255.0
    H, W, C = image.shape

    target = torch.from_numpy(image).float().cuda()
    model = Model(H=H, W=W, scale_factor=scale_factor).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-20)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.999)
    for i in trange(5000):
        net_out = model.forward()
        loss = F.mse_loss(net_out, target)
        psnr = mse2psnr(loss.detach())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-1)
        optimizer.step()
        scheduler.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-1)
        print(f"iter: {i}, loss: {loss}, psnr: {psnr}, lr: {optimizer.param_groups[0]['lr']}")

        plt.imsave(f'outputs_fixed_l/output_{i}.jpg', net_out.detach().cpu().numpy())
        # if i > 0 and i % 200:
        #     optimizer.param_groups[0]['lr'] /= 0.5
