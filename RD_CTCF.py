import numpy as np
import h5py

window_size = 512
step = 512

wt = np.r_[0.:float(window_size)]
al = np.linspace(-2.5, 2.5, num=int(window_size*2+1))

def cmf(s, p, q):

    L = s.shape[-1]
    m = p - q
    
    s = pow(s, m) * pow(np.conj(s), q)
    no = int(np.floor((L - window_size) / step)) + 1
    shape = (no, window_size)
    out = []
    for i in s:
        strides = (i.strides[-1] * step, i.strides[-1])  # size block * window
        out.append(np.lib.stride_tricks.as_strided(i, shape=shape, strides=strides))
    
    s = np.asarray(out)    # size n * block * window

    s = np.fft.fft(s)
    print (s.shape)
    s = np.mean(s, axis=1)
    s = np.fft.ifft(s)
    return s


def RD_CTCF(s):

    s = np.asarray(s)
    a = s.real
    b = s.imag
    #s = s / np.amax(np.sqrt(np.mean(a + b, axis=-1, keepdims=True)))
    M20 = cmf(s, p=2, q=0)
    M21 = cmf(s, p=2, q=1)
    M22 = cmf(s, p=2, q=2)
    M40 = cmf(s, p=4, q=0)
    M41 = cmf(s, p=4, q=1)
    M42 = cmf(s, p=4, q=2)
    M43 = cmf(s, p=4, q=3)
    M61 = cmf(s, p=6, q=1)
    M63 = cmf(s, p=6, q=3)
    
    C40 = M40 - M20*M20 - 2*M20*M20
    C42 = M42 - abs(M20)**2 - 2*M21*M21
    C61 = M61 - 5*M21*M40 - 10*M20*M41 + 30*M20*M20*M21
    C63 = M63 - 9*M21*M42  - 3*M20*M43 - 3*M22*M41 + 18*M20*M21*M22 + 12*M21*M21*M21
    
    suanzi = np.outer(np.asarray(al), np.asarray(wt))
    suanzi = np.exp(2j*np.pi*suanzi)  # size alpha * window

    cmf40 = np.einsum('ni,ji->nj', C40, suanzi) / np.asarray(window_size)
    cmf42 = np.einsum('ni,ji->nj', C42, suanzi) / np.asarray(window_size)
    cmf61 = np.einsum('ni,ji->nj', C61, suanzi) / np.asarray(window_size)
    cmf63 = np.einsum('ni,ji->nj', C63, suanzi) / np.asarray(window_size)

    return np.stack((cmf40, cmf42, cmf61, cmf63)).transpose((1, 0, 2))
