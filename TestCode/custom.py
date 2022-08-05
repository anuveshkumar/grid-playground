import torch
import matplotlib.pyplot as plt

x = torch.cat([torch.linspace(1, -1, 100), torch.linspace(-1, 1, 100)])
sdf = torch.tanh(x)
mapping = torch.linspace(0, 1, 200)


