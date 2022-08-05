import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Density(nn.Module):
    def __init__(self, param_init):
        super(Density, self).__init__()
        for p in param_init:
            param = nn.Parameter(torch.tensor(param_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)


class LaplaceDensity(Density):
    def __init__(self, param_init={'beta': 0.5}, beta_min=0.0001):
        super(LaplaceDensity, self).__init__(param_init=param_init)
        self.beta_min = torch.tensor(beta_min)

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta


if __name__ == "__main__":
    ldensity = LaplaceDensity()
    mapping = torch.linspace(0, 1, 400)

    sdf = torch.cat([torch.linspace(2, -2, 200), torch.linspace(-2, 1, 100), torch.linspace(1, -1, 100)])
    density = ldensity(sdf).detach()
    z_vals = torch.linspace(0, 6, 400)

    dists = z_vals[1:] - z_vals[:-1]
    dists = torch.cat([dists, torch.tensor([6 / 400])])

    free_energy = dists * density

    shifted_free_density = torch.cat([torch.zeros([1]), free_energy[:-1]])

    alpha = 1 - torch.exp(-free_energy)
    transmittance = torch.exp(-torch.cumsum(shifted_free_density, dim=-1))
    weights = alpha * transmittance
    weights_2 = torch.softmax(alpha / .001, dim=-1)
    print(weights.sum())
    plt.plot(mapping, sdf / sdf.abs().max(), "-b", label='sdf')
    plt.plot(mapping, density / density.abs().max(), "-r", label='density')
    plt.plot(mapping, weights / weights.abs().max(), "-g", label='weights')
    plt.plot(mapping, weights_2 / weights_2.abs().max(), "-g", label='weights_2')
    plt.legend(loc="upper left")
    plt.show()
