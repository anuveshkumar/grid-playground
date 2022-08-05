import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F


def cumprod_exclusive(p):
    return torch.cat([torch.ones_like(p[..., [0]]), p.cumprod(-1)], -1)


def get_ray_marching_ray(alpha):
    alphainv_cum = cumprod_exclusive(1 - alpha)
    weights = alpha * alphainv_cum[..., :-1]
    return weights, alphainv_cum


def activate_density(sigma, interval=0.01):
    return 1 - torch.exp(-F.relu(sigma) * interval)


sdf = torch.cat([torch.linspace(2, -2, 200), torch.linspace(-2, 1, 100), torch.linspace(1, -1, 100)])
mapping = torch.linspace(0, 1, 400)
mask_greater = sdf < 0
mask_less = sdf > 0
#
beta = 0.1
sigma = torch.where(sdf <= 0, 0.5 * torch.exp(sdf / beta), 1 - 0.5 * torch.exp(-sdf / beta))

alpha = activate_density(sigma)
weights, _ = get_ray_marching_ray(alpha)

print(max(sdf.abs().max(), sigma.abs().max(), alpha.abs().max(), weights.abs().max()))
print(weights.sum())
plt.plot(mapping, sdf / 10, "-b", label='sdf')
# plt.plot(mapping, sigma / sigma.abs().max(), "-r", label='density')
# plt.plot(mapping, alpha, "-g", label='alpha')
plt.plot(mapping, weights * 2, label='weights')
plt.plot(mapping, torch.zeros_like(weights))
plt.legend(loc="upper left")

# plt.plot(mapping, weights, label='weights')
plt.show()
