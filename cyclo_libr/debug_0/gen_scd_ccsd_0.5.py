import h5py, pickle
import numpy as np
import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LRG
from sklearn.decomposition import PCA
import lib.fot as fot
import lib.sot as crr
import lib.hos as hos
from skimage.measure import block_reduce

mods_ = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK',\
         '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', \
         '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
mod = [i for i in range(0, 24) if i not in [17,18,19,20,21]]  # all digital mods.

hf = h5py.File('../dataset/201801a_data_test.h5', 'r+')
x = hf['test']
p = int(len(x) / 24)
pp = int(p / 26)
pts = 200

fi = open("logbook.txt", "w")
fi = open("logbook.txt", "a")
fi.write("Start! \n")

def run(snr):
    scd0 = []
    scd1 = []
    ccsd0 = []
    ccsd1 = []
    for i in mod:
        for j in snr:
            base = i*p + j*pp + 2
            s = x[base:(base + pts),:,0] + 1j*x[base:(base + pts),:,1]
            scd = crr.SCD(s, N=256, L=128)
            ccsd = crr.CCSD(s, N=256, L=128, sigma=0.5)
            
            scd0.append(np.concatenate((np.amax(scd, axis=-1), np.amax(scd, axis=-2)), axis=-1))
            scd1.append(block_reduce(scd, block_size=(1, 4, 4), func=np.max).reshape((pts, -1)))
            ccsd0.append(np.concatenate((np.amax(ccsd, axis=-1), np.amax(ccsd, axis=-2)), axis=-1))
            ccsd1.append(block_reduce(ccsd, block_size=(1, 4, 4), func=np.max).reshape((pts, -1)))
            
    scd0 = np.asarray(scd0).reshape((-1, int(scd.shape[-1]*2)))
    scd1 = np.asarray(scd1).reshape((len(scd0), -1))
    ccsd0 = np.asarray(ccsd0).reshape((-1, int(ccsd.shape[-1]*2)))
    ccsd1 = np.asarray(ccsd1).reshape((len(ccsd0), -1))
    print (scd0.shape)
    print (scd1.shape)
    print (ccsd0.shape)
    print (ccsd1.shape)
    return scd0, scd1, ccsd0, ccsd1

#######################  LOW SNR.
snr = range(0,8)
x0, xx0, x1, xx1 = run(snr)
yy = []
for i in range(len(mod)):
    yy.append([i] * int(len(x1) / len(mod)))
yy = np.hstack(yy)
print (yy.shape)

fi.writelines('test scd profiles, low SNR. \n')
lda = LDA().fit(x0, yy)
sc = lda.score(x0, yy)
fi.writelines('%f \n' % sc)
fi.writelines('test scd image, low SNR. \n')
lda = LDA().fit(xx0, yy)
sc = lda.score(xx0, yy)
fi.writelines('%f \n' % sc)

fi.writelines('test ccsd profiles, low SNR. \n')
lda = LDA().fit(x1, yy)
sc = lda.score(x1, yy)
fi.writelines('%f \n' % sc)
fi.writelines('test ccsd image, low SNR. \n')
lda = LDA().fit(xx1, yy)
sc = lda.score(xx1, yy)
fi.writelines('%f \n' % sc)

#######################  MID SNR.
snr = range(8,16)
x0, xx0, x1, xx1 = run(snr)
yy = []
for i in range(len(mod)):
    yy.append([i] * int(len(x1) / len(mod)))
yy = np.hstack(yy)
print (yy.shape)

fi.writelines('test scd profiles, mid SNR. \n')
lda = LDA().fit(x0, yy)
sc = lda.score(x0, yy)
fi.writelines('%f \n' % sc)
fi.writelines('test scd image, mid SNR. \n')
lda = LDA().fit(xx0, yy)
sc = lda.score(xx0, yy)
fi.writelines('%f \n' % sc)

fi.writelines('test ccsd profiles, mid SNR. \n')
lda = LDA().fit(x1, yy)
sc = lda.score(x1, yy)
fi.writelines('%f \n' % sc)
fi.writelines('test ccsd image, mid SNR. \n')
lda = LDA().fit(xx1, yy)
sc = lda.score(xx1, yy)
fi.writelines('%f \n' % sc)

#######################  HIGH SNR.
snr = range(16,26)
x0, xx0, x1, xx1 = run(snr)
yy = []
for i in range(len(mod)):
    yy.append([i] * int(len(x1) / len(mod)))
yy = np.hstack(yy)
print (yy.shape)

fi.writelines('test scd profiles, high SNR. \n')
lda = LDA().fit(x0, yy)
sc = lda.score(x0, yy)
fi.writelines('%f \n' % sc)
fi.writelines('test scd image, high SNR. \n')
lda = LDA().fit(xx0, yy)
sc = lda.score(xx0, yy)
fi.writelines('%f \n' % sc)

fi.writelines('test ccsd profiles, high SNR. \n')
lda = LDA().fit(x1, yy)
sc = lda.score(x1, yy)
fi.writelines('%f \n' % sc)
fi.writelines('test ccsd image, high SNR. \n')
lda = LDA().fit(xx1, yy)
sc = lda.score(xx1, yy)
fi.writelines('%f \n' % sc)


hf.close()
fi.close()
