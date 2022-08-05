import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


def activate_density(density, interval=0.5):
    return 1 - torch.exp(-F.softplus(density - 4.595119850134584) * interval)


def cumprod_exclusive(p):
    # Not sure why: it will be slow at the end of training if clamping at 1e-10 is not applied
    return torch.cat([torch.ones_like(p[..., [0]]), p.clamp_min(1e-10).cumprod(-1)], -1)


def get_ray_marching_ray(alpha):
    alphainv_cum = cumprod_exclusive(1 - alpha)
    weights = alpha * alphainv_cum[..., :-1]
    return weights, alphainv_cum


density = torch.tensor([-21.5873, -21.6774, -21.7643, -21.8443, -21.9210, -21.9944, -22.0644,
                        -22.1285, -22.1880, -22.2441, -22.2974, -22.3474, -22.3939, -22.4373,
                        -22.4775, -22.5142, -22.5435, -22.5697, -22.5928, -22.5807, -22.5697,
                        -22.5592, -22.5502, -22.5428, -22.5365, -22.5325, -22.5307, -22.5313,
                        -22.5336, -22.5383, -22.5455, -22.5553, -22.5676, -23.1057, -23.7608,
                        -24.4737, -25.2420, -25.9365, -26.5999, -27.2520, -27.8756, -28.5188,
                        -29.1757, -29.8224, -30.4500, -31.0743, -31.6896, -32.2515, -32.7906,
                        -33.2977, -33.8512, -34.3967, -34.9233, -35.4251, -35.9108, -36.2447,
                        -36.3115, -36.3067, -36.1368, -35.9117, -35.5308, -35.0456, -34.4617,
                        -33.7794, -32.9552, -31.8201, -30.5611, -28.7903, -26.6647, -24.5518,
                        -22.5125, -20.5579, -18.6803, -17.0137, -16.1635, -15.4108, -14.8930,
                        -14.5599, -14.7045, -15.2669, -15.9109, -16.7444, -17.6183, -17.6585,
                        -17.4892, -16.7840, -16.1359, -15.6812, -15.3790, -15.3008, -15.4826,
                        -15.9688, -16.5360, -16.8417, -16.9542, -16.8938, -16.8699, -17.1335,
                        -17.3876, -17.7067, -17.9151, -17.9496, -17.6226, -17.3057, -16.9956,
                        -16.7014, -16.3093, -15.9646, -15.6735, -15.4770, -15.5696, -15.9666,
                        -16.4395, -17.1531, -17.8084, -16.7490, -16.5740, -17.2809, -20.7822,
                        -22.2552, -21.9833, -20.7376, -16.7366, -12.4004, -11.2769, -10.8356,
                        -10.0451, -9.0780, -8.3447, -7.7152, -7.3183, -7.1786, -7.1686,
                        -7.4420, -7.7945, -8.0494, -8.4904, -8.8899, -9.0153, -9.0302,
                        -8.9495, -9.0888, -9.2320, -9.5163, -9.8961, -11.8549, -13.9648,
                        -15.5797, -17.1028, -18.0882, -18.5745, -18.4874, -17.7848, -16.6854,
                        -14.6440, -12.6991, -11.2941, -10.3488, -9.8180, -9.5810, -9.4474,
                        -9.6892, -9.9994, -10.7973, -11.7247, -12.5031, -13.1027, -13.5973,
                        -13.8518, -14.1040, -14.1438, -14.7408, -15.2228, -15.8490, -15.1906,
                        -13.1704, -11.4577, -6.7637, -2.0788, 0.4371, 1.9528, 4.9692,
                        8.6476, 12.3354, 14.9896, 16.0915, 13.4623, 10.0182, 6.5687,
                        5.5653, 5.5728, 5.2513, 5.1401, 4.9490, 4.2590, 3.6551,
                        3.1196, 2.6105, 1.6701, 0.6895, 0.8425, 1.3702, 1.1989,
                        0.5959, 0.6631, 1.1555, 1.3674, 1.4015, 1.6808, 2.2888,
                        3.1652, 4.6444, 6.3585, 9.5355, 12.9142, 14.7941, 13.6937,
                        11.8783, 12.3586, 12.1284, 11.0001, 6.5337, 3.0771, 0.4465,
                        -4.2646, -7.1557, -5.8003, -14.9020, -23.0568, -28.3293, -28.2845,
                        -28.5461, -29.4625, -30.6450, -31.1314, -31.2608, -28.7821, -24.8816,
                        -21.4825, -18.4746, -15.0469, -14.7483, -14.2272, -16.5848, -19.5529,
                        -19.2012, -17.1470, -16.3950, -18.1014, -18.7757, -17.9085, -17.2959,
                        -17.1084, -16.0539, -15.0245, -17.5677, -20.6722, -24.2780, -27.9504,
                        -31.3684, -34.1431, -36.2071, -37.3080, -37.3459, -36.4128, -35.1978,
                        -33.8836, -32.6634, -31.4904, -30.3079, -29.1246, -28.0646, -27.1142,
                        -26.0125, -24.9861, -24.0202, -23.1341, -22.3352, -21.6556, -21.0827,
                        -20.6200, -20.2653, -20.3145, -20.5241, -20.8091, -21.1698, -21.6134,
                        -22.1571, -22.7632, -23.4326, -24.1636, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
mapping = torch.linspace(0, 1, len(density))
alpha = activate_density(density)
weights, _ = get_ray_marching_ray(alpha)

plt.plot(mapping, density / density.abs().max(), label='density')
plt.plot(mapping, alpha / alpha.abs().max(), label='alpha')
plt.plot(mapping, weights / weights.abs().max(), label='weights')
plt.legend(loc="upper left")
plt.show()