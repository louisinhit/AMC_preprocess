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

#fi = open("logbook.txt", "w")
fi = open("logbook.txt", "a")
fi.write("Start! \n")

hoc = hos.HOC_cyclic(window_size=256, step=128)

def run(snr):
    xx = []
    for i in mod:
        for j in snr:
            for k in range(2, pts + 2):
                base = i*p + j*pp + k
                s = hoc(x[base,:,0] + 1j*x[base,:,1])
                xx.append(abs(s).flatten())
    xx = np.asarray(xx)
    print (xx.shape)
    return xx

#######################  LOW SNR.
snr = range(0,8)
xx = run(snr)
yy = []
for i in range(len(mod)):
    yy.append([i] * int(len(xx) / len(mod)))
yy = np.hstack(yy)
print (yy.shape)

fi.writelines('test rd ctcf ws 256 st 16 out len 512, low SNR. \n')
lda = LDA().fit(xx, yy)
sc = lda.score(xx, yy)
fi.writelines('%f \n' % sc)
#######################  MID SNR.
snr = range(8,16)
xx = run(snr)
yy = []
for i in range(len(mod)):
    yy.append([i] * int(len(xx) / len(mod)))
yy = np.hstack(yy)
print (yy.shape)

fi.writelines('test rd ctcf ws 256 st 16 out len 512, mid SNR. \n')
lda = LDA().fit(xx, yy)
sc = lda.score(xx, yy)
fi.writelines('%f \n' % sc)
#######################  HIGH SNR.
snr = range(16,26)
xx = run(snr)
yy = []
for i in range(len(mod)):
    yy.append([i] * int(len(xx) / len(mod)))
yy = np.hstack(yy)
print (yy.shape)

fi.writelines('test rd ctcf ws 256 st 16 out len 512, high SNR. \n')
lda = LDA().fit(xx, yy)
sc = lda.score(xx, yy)
fi.writelines('%f \n' % sc)

hf.close()
fi.close()
