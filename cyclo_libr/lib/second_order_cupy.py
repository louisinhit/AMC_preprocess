import numpy as np
import cupy as cp
import math
from scipy import signal
import os, sys, time


def block_samples(s, window_size, step):
    L = s.shape[-1]
    no = int(np.floor((L - window_size) / step)) + 1
    shape = (no, window_size)
    out = []
    for i in s:
        strides = (i.strides[-1] * step, i.strides[-1])  # size block * window
        out.append(np.lib.stride_tricks.as_strided(i, shape=shape, strides=strides))
    return np.asarray(out)


def scf_fam(s, ws=256):
    '''
    I follow the wiki step by step to write this function:
    https://en.wikipedia.org/wiki/Spectral_correlation_density
    '''
    L = s.shape[-1]
    step = int(ws // 4)    
    no = int(np.floor((L - ws) / step)) + 1
    Pe = int(np.floor(int(np.log(no)/np.log(2))))
    P = 2**(Pe+1)
    s = np.concatenate((s, np.zeros((s.shape[0], int((P-1)*step+ws)-L), dtype=s.dtype)), axis=-1)

    s = block_samples(s, ws, step)
    P = s.shape[1]
    window = signal.windows.hamming(ws)
    window /= np.sqrt(np.sum(window**2))
    s = cp.asarray(s * window)
    s = cp.fft.fftshift(cp.fft.fft(s, axis=-1), axes=(-1))
    
    omega = cp.arange(ws) / float(ws) - 0.5
    suanzi = cp.outer(cp.arange(P), omega)
    suanzi = cp.exp(-2j*cp.pi*step*suanzi)

    s = cp.multiply(s, suanzi)
    scf = cp.einsum('mij,mik->mijk', cp.conjugate(s), s)
    
    return cp.asnumpy(cp.abs(cp.average(scf, axis=1)))


def CHTC(s: np.ndarray, ws: int=512, step: int=512) -> np.ndarray:
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
    return cp.asnumpy(o)


def CCSD(s: np.ndarray, ws: int=512, step: int=512, sigma: float=0.3) -> np.ndarray:
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
    