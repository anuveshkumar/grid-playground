import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import itertools
import time


class PREF(nn.Module):
    def __init__(self, res=[160] * 3, d=[8] * 3, ks=16, device='cuda'):
        super(PREF, self).__init__()
        # res: resolution size
        # d: reduce dim size
        # ks: output kernel size
        self.device = device
        self.res = res
        self.d = d
        self.output_dim = ks
        Nx, Ny, Nz = res
        dx, dy, dz = d
        # log sampling freq in reduced dimension
        self.freq = [torch.tensor([0.] + [2 ** i for i in torch.arange(dim - 1)]).to(self.device) for dim in
                     d]
        self.alpha_params = nn.Parameter(torch.tensor([1e-3]).to(self.device))
        self.params = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, ks, dx, Ny, Nz).to(torch.complex64).to(self.device)),
             nn.Parameter(torch.zeros(1, ks, Nx, dy, Nz).to(torch.complex64).to(self.device)),
             nn.Parameter(torch.zeros(1, ks, Nx, Ny, dz).to(torch.complex64).to(self.device))
             ]
            # self.init_phasor_volume()
        )
        torch.cuda.empty_cache()
        self.ktraj = self.compute_ktraj(self.freq, res=res)

    @property
    def alpha(self):
        # adaptively adjust the scale of phasors' magnitude during optimization.
        # not so important when Parsvel loss is imposed.
        return F.softplus(self.alpha_params, beta=10, threshold=1)

    @property
    def phasor(self):
        feature = [feat * self.alpha for feat in self.params]
        return feature

    def forward(self, xyz, bound, interp=False):
        xyz = xyz / bound
        if interp:
            fx, fy, fz = self.compute_spatial_volume(self.phasor)
            volume = fx + fy + fz
            P = F.grid_sample(volume, xyz[None, None, None].flip(-1), align_corners=True, mode='bilinear') \
                .view(-1, *xyz.shape[:1])
            return P.T
        else:
            # 2D FFT
            Pu, Pv, Pw = self.phasor
            Pu = torch.fft.ifftn(Pu, dim=(3, 4))
            Pv = torch.fft.ifftn(Pv, dim=(2, 4))
            Pw = torch.fft.ifftn(Pw, dim=(2, 3))

            # 2D Linear Interpolation
            xs, ys, zs = xyz.chunk(3, dim=-1)
            Px = grid_sample_cmplx(Pu.transpose(3, 3).flatten(1, 2), torch.stack([zs, ys], dim=-1)[None]).reshape(
                Pu.shape[1], Pu.shape[2], -1)
            Py = grid_sample_cmplx(Pv.transpose(2, 3).flatten(1, 2), torch.stack([zs, xs], dim=-1)[None]).reshape(
                Pv.shape[1], Pv.shape[3], -1)
            Pz = grid_sample_cmplx(Pw.transpose(2, 4).flatten(1, 2), torch.stack([xs, ys], dim=-1)[None]).reshape(
                Pw.shape[1], Pw.shape[4], -1)
            fx, fy, fz = self.freq
            Pux = batch_irfft(Px, xs, fx)
            Pvy = batch_irfft(Py, ys, fy)
            Pwz = batch_irfft(Pz, zs, fz)
            P = Pux + Pvy + Pwz
            return P.T

    def compute_ktraj(self, axis, res):  # the associated frequency coordinates.
        ktraj2d = [torch.fft.fftfreq(i, 1 / i).to(self.device) for i in res]
        ktraj1d = [torch.arange(ax).to(torch.float).to(self.device) if type(ax) == int else ax for ax in axis]
        ktrajx = torch.stack(torch.meshgrid([ktraj1d[0], ktraj2d[1], ktraj2d[2]]), dim=-1)
        ktrajy = torch.stack(torch.meshgrid([ktraj2d[0], ktraj1d[1], ktraj2d[2]]), dim=-1)
        ktrajz = torch.stack(torch.meshgrid([ktraj2d[0], ktraj2d[1], ktraj1d[2]]), dim=-1)
        ktraj = [ktrajx, ktrajy, ktrajz]
        return ktraj

    def compute_spatial_volume(self, features):
        xx, yy, zz = [torch.linspace(0, 1, N).to(self.device) for N in self.res]
        Pu, Pv, Fz = features
        Nx, Ny, Nz = Pv.shape[2], Fz.shape[3], Pu.shape[4]
        d1, d2, d3 = Pu.shape[2], Pv.shape[3], Fz.shape[4]
        kx, ky, kz = self.freq
        kx, ky, kz = kx[:d1], ky[:d2], kz[:d3]
        fx = irfft(torch.fft.ifftn(Pu, dim=(3, 4)), xx, ff=kx, T=Nx, dim=2)
        fy = irfft(torch.fft.ifftn(Pv, dim=(2, 4)), yy, ff=ky, T=Ny, dim=3)
        fz = irfft(torch.fft.ifftn(Fz, dim=(2, 3)), zz, ff=kz, T=Nz, dim=4)
        return (fx, fy, fz)

    def parseval_loss(self):
        # Parseval Loss
        new_feats = [Fk.reshape(-1, *Fk.shape[2:], 1) * 1j * np.pi * wk.reshape(1, *Fk.shape[2:], -1)
                     for Fk, wk in zip(self.phasor, self.ktraj)]
        loss = sum([feat.abs().square().mean() for feat in itertools.chain(*new_feats)])
        return loss

    @torch.no_grad()
    def init_phasor_volume(self):
        # rough approximation
        # transform the fourier domain to spatial domain
        Nx, Ny, Nz = self.res
        d1, d2, d3 = self.d
        # xx, yy, zz = [torch.linspace(0, 1, N).to(self.device) for N in (d1,d2,d3)]
        xx, yy, zz = [torch.linspace(0, 1, N).to(self.device) for N in (Nx, Ny, Nz)]
        XX, YY, ZZ = [torch.linspace(0, 1, N).to(self.device) for N in (Nx, Ny, Nz)]
        kx, ky, kz = self.freq
        kx, ky, kz = kx[:d1], ky[:d2], kz[:d3]

        fx = torch.ones(1, self.output_dim, len(xx), Ny, Nz).to(self.device)
        fy = torch.ones(1, self.output_dim, Nx, len(yy), Nz).to(self.device)
        fz = torch.ones(1, self.output_dim, Nx, Ny, len(zz)).to(self.device)
        normx = torch.stack(torch.meshgrid([2 * xx - 1, 2 * YY - 1, 2 * ZZ - 1]), dim=-1).norm(dim=-1)
        normy = torch.stack(torch.meshgrid([2 * XX - 1, 2 * yy - 1, 2 * ZZ - 1]), dim=-1).norm(dim=-1)
        normz = torch.stack(torch.meshgrid([2 * XX - 1, 2 * YY - 1, 2 * zz - 1]), dim=-1).norm(dim=-1)

        fx = fx * normx[None, None] / (3 * self.alpha * np.sqrt(self.output_dim))
        fy = fy * normy[None, None] / (3 * self.alpha * np.sqrt(self.output_dim))
        fz = fz * normz[None, None] / (3 * self.alpha * np.sqrt(self.output_dim))

        fxx = rfft(torch.fft.fftn(fx.transpose(2, 4), dim=(2, 3), norm='forward'), xx, ff=kx, T=Nx).transpose(2, 4)
        fyy = rfft(torch.fft.fftn(fy.transpose(3, 4), dim=(2, 3), norm='forward'), yy, ff=ky, T=Ny).transpose(3, 4)
        fzz = rfft(torch.fft.fftn(fz.transpose(4, 4), dim=(2, 3), norm='forward'), zz, ff=kz, T=Nz).transpose(4, 4)
        return [torch.nn.Parameter(fxx), torch.nn.Parameter(fyy), torch.nn.Parameter(fzz)]


def grid_sample_cmplx(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True):
    sampled = F.grid_sample(input.real, grid, mode, padding_mode, align_corners) + \
              1j * F.grid_sample(input.imag, grid, mode, padding_mode, align_corners)
    return sampled


def irfft(phasors, xx, ff=None, T=None, dim=-1):
    assert (xx.max() <= 1) & (xx.min() >= 0)
    phasors = phasors.transpose(dim, -1)
    assert phasors.shape[-1] == len(ff) if ff is not None else True
    device = phasors.device
    # xx = xx * (T - 1)  # / T  # to match torch.fft.fft
    N = phasors.shape[-1]
    if ff is None:
        ff = torch.arange(N).to(device)  # positive freq only
    xx = xx.reshape(-1, 1).to(device)
    M = torch.exp(2j * np.pi * xx * ff).to(device)
    M = M * ((ff > 0) + 1)[None]  # Hermittion symmetry
    out = F.linear(phasors.real, M.real) - F.linear(phasors.imag, M.imag)
    out = out.transpose(dim, -1)
    return out


def batch_irfft(phasors, xx, ff):
    # numerial integration
    # phaosrs [dim, d, N] # coords  [N,1] # bandwidth d  # norm x to [0,1]
    xx = (xx + 1) * 0.5
    if ff is None:
        ff = torch.arange(phasors.shape[1]).to(xx.device)
    twiddle = torch.exp(2j * np.pi * xx * ff)  # twiddle factor
    twiddle = twiddle * ((ff > 0) + 1)[None]  # hermitian # [N, d]
    twiddle = twiddle.transpose(0, 1)[None]
    return (phasors.real * twiddle.real).sum(1) - (phasors.imag * twiddle.imag).sum(1)


@torch.no_grad()
def rfft(spatial, xx, ff=None, T=None, dim=-1):
    assert (xx.max() <= 1) & (xx.min() >= 0)
    spatial = spatial.transpose(dim, -1)
    assert spatial.shape[-1] == len(xx)
    device = spatial.device
    xx = xx * (T - 1) / T
    if ff is None:
        ff = torch.fft.rfftfreq(T, 1 / T)  # positive freq only
    ff = ff.reshape(-1, 1).to(device)
    M = torch.exp(-2j * np.pi * ff * xx).to(device)
    out = F.linear(spatial, M)
    out = out.transpose(dim, -1) / len(xx)
    return out


if __name__ == "__main__":
    from tqdm import trange

    encoder = PREF(res=[128] * 3, d=[6] * 3, ks=16)
    model = nn.Sequential(
        nn.Linear(16, 64), nn.ReLU(inplace=True),
        nn.Linear(64, 64), nn.ReLU(inplace=True),
        nn.Linear(64, 1)
    ).cuda()

    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(model.parameters()), lr=2e-3)

    for i in trange(1000):
        timeit = time.time()
        x = torch.FloatTensor(4096 * 500, 3).uniform_(-1.4, 1.4).cuda()
        # print(i)

        # timeit = time.time()
        out1 = encoder.forward(x, bound=1.4, interp=True)
        out2 = encoder.forward(x, bound=1.4, interp=False)
        # print(time.time() - timeit)

        # timeit = time.time()
        net_out = model.forward(out1)
        gt = torch.rand_like(net_out)
        loss = F.l1_loss(net_out, gt)
        # print(time.time() - timeit)

        # timeit = time.time()
        loss.backward()
        optimizer.step()
        # print(time.time() - timeit)

    # print(out)
    # print(out.shape)
