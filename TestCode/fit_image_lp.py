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


class Model(nn.Module):
    def __init__(self, H, W, latent_size=4, channel_size=16, kernel_size=3, padding=1):
        super(Model, self).__init__()
        self.conv1_1 = nn.Conv2d(latent_size, channel_size, kernel_size=kernel_size, stride=1, padding=padding,
                                 bias=True)
        self.conv1_2 = nn.Conv2d(channel_size + latent_size, channel_size, kernel_size=kernel_size, stride=1,
                                 padding=padding, bias=True)
        self.conv2 = nn.Conv2d(channel_size, channel_size * 4, kernel_size=kernel_size, stride=1, padding=padding,
                               bias=True)

        self.pixel_shuffle = nn.PixelShuffle(2)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(channel_size + latent_size + 3, channel_size, kernel_size=kernel_size, stride=1, padding=padding,
                      bias=True),
            nn.BatchNorm2d(channel_size),
            nn.ReLU(),
            nn.Conv2d(channel_size, channel_size, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(channel_size),
            nn.ReLU()
        )

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.translate_1 = nn.Conv2d(channel_size, 3, kernel_size=kernel_size, stride=1, padding=padding, bias=True)
        self.translate_2 = nn.Conv2d(channel_size, 3, kernel_size=kernel_size, stride=1, padding=padding, bias=True)

        self.bn1_1 = nn.BatchNorm2d(channel_size)
        self.bn1_2 = nn.BatchNorm2d(channel_size)
        self.bn2 = nn.BatchNorm2d(channel_size * 4)

        self.act_fn_1 = nn.ReLU()
        self.act_fn_2 = nn.ReLU()

        self.grid = torch.nn.Parameter(torch.zeros(1, latent_size, H // 2, W // 2))

    def forward(self):
        x = self.act_fn_1(self.bn1_1(self.conv1_1(self.grid)))
        # x = torch.cat((x, self.grid), dim=1)
        # x = self.act_fn_1(self.bn1_2(self.conv1_2(x)))
        rgb_1 = self.translate_1(x)
        x = torch.cat((x, self.grid, rgb_1), dim=1)
        x = self.feature_extractor(x)
        x = self.act_fn_2(self.bn2(self.conv2(x)))
        x = self.pixel_shuffle(x)
        rgb_2 = self.translate_2(x) + self.upsample(rgb_1)
        # plt.imshow(torch.sigmoid(self.translate_2(x)).squeeze().detach().cpu().permute(1, 2, 0))
        out = [torch.sigmoid(i) for i in [rgb_1, rgb_2]]

        return out


mse2psnr = lambda x: -10. * torch.log10(x)
if __name__ == "__main__":
    seed_everything()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    os.makedirs('outputs_lp', exist_ok=True)
    image = np.array(Image.open('/home/architect/data/nerf_synthetic/drums/train/r_0.png'))[..., :3] / 255.0
    H, W, C = image.shape

    target = [cv2.resize(image, (H // x, W // x), interpolation=cv2.INTER_CUBIC) for x in [2, 1]]
    target = [torch.from_numpy(i).permute(2, 0, 1).unsqueeze(0).float().cuda() for i in target]
    model = Model(H=H, W=W).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
    for i in trange(1200):
        psnrs = []
        losses = []
        net_out = model.forward()
        total_loss = torch.tensor([0.])
        for j, (out, tar) in enumerate(zip(net_out, target)):
            # if j == 0:
            #     continue
            loss = F.mse_loss(out, tar)
            total_loss += loss
            losses.append(loss.detach())
            if j == 0:
                psnrs.append(mse2psnr(F.mse_loss(model.upsample(out).detach(), target[-1])))
            else:
                psnrs.append(mse2psnr(F.mse_loss(out.detach(), target[-1])))

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-1)
        optimizer.step()
        scheduler.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1e-1)
        print(f"iter: {i}, loss: {losses}, psnr: {psnrs}, lr: {optimizer.param_groups[0]['lr']}")

        plt.imsave(f'outputs_lp/output_{i}_.jpg',
                   net_out[-1].squeeze().permute(1, 2, 0).detach().cpu().clip_(0, 1).numpy())
        # if i > 0 and i % 200:
        #     optimizer.param_groups[0]['lr'] /= 0.5
