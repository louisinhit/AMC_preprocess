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


def SCD(s: np.ndarray, ws: int, step: int=0, padding: bool=True, mtd: int=2) -> np.ndarray:
    '''
    I follow the wiki step by step to write this function:
    https://en.wikipedia.org/wiki/Spectral_correlation_density
    '''
    L = s.shape[-1]
    N = s.shape[0]

    if step == 0:
        step = ws // 4
    
    no = int(np.floor((L - ws) / step)) + 1

    if no & (no - 1) != 0 and padding:
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

    if mtd == 0:
        return cp.asnumpy(cp.abs(cp.average(scf, axis=1)))

    elif mtd == 1:
        scf = cp.fft.fftshift(cp.fft.fft(scf, axis=1), axes=1)
        return cp.asnumpy(np.abs(cp.average(scf, axis=1)))
    
    elif mtd == 2:
    ############################
    ####### original FAM #######
    ############################
        scf = cp.fft.fftshift(cp.fftfft(scf, axis=1), axes=1)
        scf = cp.abs(scf)
        Sx = cp.zeros((N, ws, 2*P*step), dtype=float)
        Mp = (P*step)//ws//2
        for k in range(ws):
            for l in range(ws):
                i = int((k+l)/2.0)
                a = int(((k-l)/float(ws)+1.)*P*step)
                Sx[:, i, a-Mp:a+Mp] = scf[:, (P//2-Mp):(P//2+Mp),k,l]
        return cp.asnumpy(Sx)


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
    
