import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# class SingleVarianceNetwork(nn.Module):
#     def __init__(self, init_val=0.3):
#         super(SingleVarianceNetwork, self).__init__()
#         self.register_parameter('variance', nn.Parameter(torch.tensor(init_val)))
#
#     def forward(self, x):
#         return torch.ones([len(x), 1]) * torch.exp(self.variance * 10.0)
#

def get_smart_weights_from_sdf(sdf, scale):
    s_density = sample_logistic_cumulative_density_distribution(sdf, scale)
    alpha = torch.clip((s_density[..., :-1] - s_density[..., 1:]) / (s_density[..., :-1] + 1e-10), 0)  # prevent nan
    acc_transmittance = torch.cumprod((1 - alpha), -1)
    bg_lambda = acc_transmittance[..., -1]
    acc_transmittance = torch.roll(acc_transmittance, 1, -1)
    acc_transmittance[..., :1] = 1
    weights = acc_transmittance * alpha

    return weights, bg_lambda


def get_naive_weights_from_sdf(sdf, scale):
    s_density_pdf = sample_logistic_probability_density_distribution(sdf, scale)
    weights = torch.exp(-torch.cumsum(s_density_pdf, -1)) * s_density_pdf
    return weights


def sample_logistic_probability_density_distribution(location, scale=torch.tensor([1.0])):
    dist = scale * torch.exp(-location * scale) / torch.pow((1 + torch.exp(-location * scale)), 2)
    dist[torch.isnan(dist)] = 1e-10
    return dist


def sample_logistic_cumulative_density_distribution(location, scale=1.0):
    return 1 / (1 + torch.exp(-location * scale))


sdf = torch.cat([torch.linspace(2, -1, 300), torch.linspace(-1, 1, 200), torch.linspace(1, -1, 100)])
weights, _ = get_smart_weights_from_sdf(sdf, scale=torch.exp(torch.tensor([0.1]) * 10))

mapping = torch.linspace(0, 1, 599)
plt.plot(mapping, sdf[:-1] / sdf.abs().max(), "-b", label='sdf')
plt.plot(mapping, weights / weights.abs().max(), "-r", label='density')
plt.legend(loc="upper left")
plt.show()
