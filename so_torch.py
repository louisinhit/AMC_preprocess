import math, torch
import sys
from torch.fft import fft, fftshift
import numpy as np


def block_samples(s, window_size, step):
    L = s.shape[-1]
    no = int(np.floor((L - window_size) / step)) + 1
    return torch.as_strided(s, (s.shape[0], no, window_size), (L, step, 1))


def scf_fam(s, ws):
    '''
    I follow the wiki step by step to write this function:
    https://en.wikipedia.org/wiki/Spectral_correlation_density
    '''
    L = s.shape[-1]
    step = int(ws // 4)    
    no = int(np.floor((L - ws) / step)) + 1
    Pe = int(np.floor(int(np.log(no)/np.log(2))))
    P = 2**(Pe+1)
    s = torch.cat((s, torch.zeros((s.shape[0], int((P-1)*step+ws)-L)).type_as(s)), dim=-1)

    s = block_samples(s, ws, step)
    P = s.shape[1]
    window = torch.hamming_window(ws).type_as(s)
    window /= torch.sqrt(torch.sum(window**2))
    s = fftshift(fft(s * window, dim=-1), dim=(-1))

    omega = torch.arange(ws) / float(ws) - 0.5
    suanzi = torch.outer(torch.arange(P), omega).type_as(s)
    suanzi = torch.exp(-2j*torch.Tensor([math.pi]).type_as(s)*step*suanzi)

    s = torch.mul(s, suanzi)
    scf = torch.einsum('mij,mik->mijk', torch.conj(s), s)
    
    return torch.abs(torch.mean(scf, dim=1))


def CCSD(s, ws, step, sigma):
    '''
    I follow the steps listed in this paper:
    https://www.sciencedirect.com/science/article/pii/S0957417416305607
    '''
    s = block_samples(s, ws, step)
    L = s.shape[0]
    n = torch.cat((s[:, 1:, :], torch.zeros((L, 1, ws)).type_as(s)), dim=1)
    s = torch.cat((s, n), dim=-1)
    o = []
    for tau in range(ws):
        c = s[:, :, :ws] - s[:, :, tau:(tau+ws)]
        x = (1 / (2*torch.Tensor([math.pi]).type_as(s)*sigma**2)) * torch.exp(-1 * c * torch.conj(c) / (2*sigma**2))
        o.append(x)

    o = torch.cat(o).view(ws, L, s.shape[-2], ws).transpose(0, 1)
    # batch, tau, block, ws
    o = torch.mean(o, dim=-2) - torch.mean(o, dim=(1, 2, 3)).view(L, 1, 1)
    o = fftshift(fft(o, dim=-1), dim=-1)
    o = fftshift(fft(o, dim=1), dim=1)
    o = torch.abs(o)
    # normalize
    mi = torch.amin(o, dim=(1, 2), keepdims=True)
    o = (o - mi) / (torch.amax(o, dim=(1, 2), keepdims=True) - mi)
    return o


def CHTC(s, ws, step):
    s = block_samples(s, ws, step)
    L = s.shape[0]
    n = torch.cat((s[:, 1:, :], torch.zeros((L, 1, ws)).type_as(s)), dim=1)
    s = torch.cat((s, n), dim=-1)
    o = []
    for tau in range(ws):
        c = torch.tanh(s[:, :, :ws] * s[:, :, tau:(tau+ws)])
        o.append(torch.abs(c))

    o = torch.cat(o).view(ws, L, s.shape[-2], ws).transpose(0, 1)
    # batch, tau, block, ws
    o = torch.mean(o, dim=-2) - torch.mean(o, dim=(1, 2, 3)).view(L, 1, 1)
    o = fftshift(fft(o, dim=-1), dim=-1)
    o = fftshift(fft(o, dim=1), dim=1)
    o = torch.abs(o)
    # normalize
    mi = torch.amin(o, dim=(1, 2), keepdims=True)
    o = (o - mi) / (torch.amax(o, dim=(1, 2), keepdims=True) - mi)
    return o
