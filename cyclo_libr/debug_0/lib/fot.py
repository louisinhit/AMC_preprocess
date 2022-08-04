import numpy as np
import numpy
import pywt
from scipy.fft import fft, fftfreq, fftshift
import scipy.signal as scp
from stockwell import st


def Raw_IQ(signal):
    return np.stack([signal.real, signal.imag], axis=1)

def FT(s, axis=-1):
    return fftshift(fft(s, axis=axis))

def CWT(s, wavename='gaus1', scal=4):
    ''' suggest use the last row of output, as it corresponds to the largest scale factor.
    '''
    Fc = pywt.central_frequency(wavename)
    Fc = 2 * Fc * scal
    scal = Fc / np.arange(1, scal + 1)

    coef, _ = pywt.cwt(s, scal, wavename, axis=-1)
    return coef.transpose((1, 0, 2))

def DWT(s, wavename='haar'):
    aa, dd = pywt.dwt(s, wavename, axis=-1)
    return np.concatenate((aa, dd), axis=-1)

def STFT(s):
    _, __, data = scp.stft(s, nperseg=256, noverlap=128, return_onesided=False)
    return data

def ST(s, fm=64):
    s = st.st(s, 0, fm)
    return s
