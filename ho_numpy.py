import numpy as np


def moment(seq, p, q):
    m = p - q
    o = pow(seq,m) * pow(np.conj(seq),q)
    return np.mean(o, axis=-1)

def normalize(C20,C21,C40,C41,C42,C60,C61,C62,C63,C80,C82,C84,M101,M103):
    C40_norm = pow(abs(C40), 1/2)
    C41_norm = pow(abs(C41), 1/2)
    C42_norm = pow(abs(C42), 1/2)
    C60_norm = pow(abs(C60), 1/3)
    C61_norm = pow(abs(C61), 1/3)
    C62_norm = pow(abs(C62), 1/3)
    C63_norm = pow(abs(C63), 1/3)
    C80_norm = pow(abs(C80), 1/4)
    C82_norm = pow(abs(C82), 1/4)
    C84_norm = pow(abs(C84), 1/4)
    M101 = pow(abs(M101), 1/5)
    M103 = pow(abs(M103), 1/5)
    return np.asarray([C20, C21, C40_norm, C41_norm, C42_norm,
                       C60_norm, C61_norm, C62_norm, C63_norm, C80_norm,
                       C82_norm, C84_norm, M101, M103]).transpose((1, 0))

def element_HOS(s):
    s = np.asarray(s)
    a = s.real
    b = s.imag
    s = s / np.sqrt(np.mean(a**2 + b**2, axis=-1, keepdims=True))

    M20 = moment(s, 2, 0)
    M21 = moment(s, 2, 1)
    M22 = moment(s, 2, 2)
    M40 = moment(s, 4, 0)
    M41 = moment(s, 4, 1)
    M42 = moment(s, 4, 2)
    M43 = moment(s, 4, 3)
    M60 = moment(s, 6, 0)
    M61 = moment(s, 6, 1)
    M62 = moment(s, 6, 2)
    M63 = moment(s, 6, 3)
    M80 = moment(s, 8, 0)
    M82 = moment(s, 8, 2)
    M84 = moment(s, 8, 4)
    M101 = moment(s, 10, 1)
    M103 = moment(s, 10, 3)
    
    C20 = M20
    C21 = M21
    C40 = M40 - 3*M20*M20
    C41 = M41 - 3*M20*M21
    C42 = M42 - abs(M20)**2 - 2*M21*M21
    C60 = M60 - 15*M20*M40 + 3*M20*M20*M20
    C61 = M61 - 5*M21*M40 - 10*M20*M41 + 30*M20*M20*M21
    C62 = M62 - 6*M20*M42 - 8*M21*M41 - M22*M40 + 6*M20*M20*M22 + 24*M21*M21*M20
    C63 = M63 - 9*M21*M42  - 3*M20*M43 - 3*M22*M41 + 18*M20*M21*M22 + 12*M21*M21*M21
    C80 = M80 - 35*M40*M40 - 630*pow(M20,4) + 420*M20*M20*M40
    C82 = M82 - 15*M40*M42 - 20*M41*M41 + 30*M40*M20*M20 + 60*M40*M21*M21 + \
          240*M41*M21*M20 + 90*M42*M20*M20 - 90*pow(M20,4) - 540*M20*M20*M21*M21
    C84 = M84 - M40**2 - 18*M42**2 - 16*M41**2 - 54*pow(M20,4) - 144*pow(M21,4) - \
          432*M20*M20*M21*M21 + 12*M40*M20**2 + 96*M41*M21*M20 + 144*M42*M21**2 + \
          72*M42*M20**2 + 96*M41*M20*M21
    return normalize(C20,C21,C40,C41,C42,C60,C61,C62,C63,C80,C82,C84,M101,M103)


def cmf(s, p, q, window_size=512, step=128):
    wt = np.r_[0.0:float(window_size)] / window_size
    al = np.linspace(-0.5*window_size, 0.5*window_size, num=int(window_size*2))
    m = p - q
    L = s.shape[-1]
    s = pow(s, m) * pow(np.conj(s), q)
    no = int(np.floor((L - window_size) / step)) + 1
    shape = (no, window_size)
    out = []
    for i in s:
        strides = (i.strides[-1] * step, i.strides[-1])  # size block * window
        out.append(np.lib.stride_tricks.as_strided(i, shape=shape, strides=strides))
    s = np.asarray(out)    # size n * block * window
        
    suanzi = np.outer(al, wt)
    suanzi = np.exp(-2j*np.pi*suanzi)  # size alpha * window
    cmf = np.einsum('nki,ji->nkj', np.asarray(s), suanzi) / np.asarray(window_size)        
    del s, suanzi, out
    return np.mean(cmf, axis=1)

    
def RD_CTCF(s):        
    s = np.asarray(s)
    a = s.real
    b = s.imag
    s = s / np.sqrt(np.mean(a**2 + b**2, axis=-1, keepdims=True))
    M20 = cmf(s, p=2, q=0)
    M21 = cmf(s, p=2, q=1)
    M22 = cmf(s, p=2, q=2)
    M40 = cmf(s, p=4, q=0)
    M41 = cmf(s, p=4, q=1)
    M42 = cmf(s, p=4, q=2)
    M43 = cmf(s, p=4, q=3)
    M61 = cmf(s, p=6, q=1)
    M63 = cmf(s, p=6, q=3)
    
    C40 = M40 - 3*M20*M20
    C42 = M42 - abs(M20)**2 - 2*M21*M21
    C61 = M61 - 5*M21*M40 - 10*M20*M41 + 30*M20*M20*M21
    C63 = M63 - 9*M21*M42  - 3*M20*M43 - 3*M22*M41 + 18*M20*M21*M22 + 12*M21*M21*M21
    return np.stack((C40, C42, C61, C63)).transpose((1, 0, 2))
    
