import numpy as np
import math
import sys
from scipy.fft import fft, fftshift
import cupy as cp
import scipy as scp


def block_samples(s, window_size, step):
    L = s.shape[-1]
    no = int(np.floor((L - window_size) / step)) + 1
    shape = (no, window_size)
    out = []
    for i in s:
        strides = (i.strides[-1] * step, i.strides[-1])  # size block * window
        out.append(np.lib.stride_tricks.as_strided(i, shape=shape, strides=strides))
    return np.asarray(out)


def CHTC(s: np.ndarray, ws: int, step: int, d: int) -> np.ndarray:
    s = block_samples(s, ws, step)
    L = s.shape[0]
    n = np.concatenate((s[:, 1:, :], np.zeros((L, 1, ws), dtype=s.dtype)), axis=1)
    s = np.concatenate((s, n), axis=-1)
    s = cp.asarray(s)
    o = []
    for tau in range(ws):
        c = s[:, :, :ws] * s[:, :, tau:(tau+ws)]
        o.append(cp.tanh(cp.abs(c)))
    
    # tau, batch, block, ws
    o = cp.asarray(o).transpose((1, 0, 2, 3))
    # batch, tau, block, ws
    o = cp.mean(o, axis=-2) - cp.expand_dims(cp.mean(o, axis=(1, 2, 3)), axis=(1, 2))
    o = cp.fft.fftshift(cp.fft.fft(o, axis=-1), axes=-1)
    o = cp.fft.fftshift(cp.fft.fft(o, axis=1), axes=1)
    o = cp.abs(o)
    # normalize
    mi = cp.amin(o, axis=(1, 2), keepdims=True)
    o = (o - mi) / (cp.amax(o, axis=(1, 2), keepdims=True) - mi)
    # svd compress
    out = []
    o = cp.asnumpy(o)
    for i in o:
        u, _, __ = scp.linalg.svd(i.T, full_matrices=True)
        out.append(np.einsum('jk,kl->jl', i, u[:, :d]))
    del o
    return np.asarray(out)


def CCSD(s: np.ndarray, ws: int, step: int, sigma: float) -> np.ndarray:
    '''
    I follow the steps listed in this paper:
    https://www.sciencedirect.com/science/article/pii/S0957417416305607
    '''
    s = block_samples(s, ws, step)
    L = s.shape[0]
    n = np.concatenate((s[:, 1:, :], np.zeros((L, 1, ws), dtype=s.dtype)), axis=1)
    s = np.concatenate((s, n), axis=-1)
    s = cp.asarray(s)
    o = []
    for tau in range(ws):
        c = s[:, :, :ws] - s[:, :, tau:(tau+ws)]
        x = (1 / (2*cp.pi*sigma**2)) * cp.exp(-1 * c * cp.conjugate(c) / (2*sigma**2))
        o.append(x)
    
    # tau, batch, block, ws
    o = cp.asarray(o).transpose((1, 0, 2, 3))
    # batch, tau, block, ws
    o = cp.mean(o, axis=-2) - cp.expand_dims(cp.mean(o, axis=(1, 2, 3)), axis=(1, 2))
    o = cp.fft.fftshift(cp.fft.fft(o, axis=-1), axes=-1)
    o = cp.fft.fftshift(cp.fft.fft(o, axis=1), axes=1)
    o = cp.abs(o)
    # normalize
    mi = cp.amin(o, axis=(1, 2), keepdims=True)
    o = (o - mi) / (cp.amax(o, axis=(1, 2), keepdims=True) - mi)
    return cp.asnumpy(o)
