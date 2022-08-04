import numpy as np
import random
import h5py, pickle, sys, argparse
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sn
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LRG
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from skimage.measure import block_reduce
sys.path.append('lib/')
import lib_interface as lib
import second_order_cupy as cyc
import feature_selection_reduce as fsr
import matplotlib
matplotlib.use('Agg')


mod_list = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK',\
         '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', \
         '256QAM', 'GMSK', 'OQPSK']
mod = [i for i in range(0, 24) if i not in [17,18,19,20,21]]  # all digital mods.

hf = h5py.File('../dataset/201801a_data_test.h5', 'r+')
x = hf['test']
p = int(len(x) / 24)
pp = int(p / 26)
pts = 300
tr = int(pts // 10) * 9
te = pts - tr

parser = argparse.ArgumentParser()
parser.add_argument('--sig', type=float, default=0.3)
args = parser.parse_args()
sigma = args.sig

print (sigma)
fi = open("pics/logbook_ccsd_{}.txt".format(sigma), "w")
fi = open("pics/logbook_ccsd_{}.txt".format(sigma), "a")
fi.write("Start \n")

def create_label(cls, num):
    mods = range(cls)
    mo = []
    for m in mods:
        mo.append([m] * num)
    mo = np.hstack(mo)
    return mo

def image(label, cm):
    fig = plt.figure()
    name = "pics/predict_ccsd_{}_SNR_{}".format(label, sigma)
    plt.title(name, fontsize =20)
    sn.set(font_scale=1.4)
    sn.heatmap(cm, cmap='Greens', xticklabels=mod_list, yticklabels=mod_list)
    # annot=True, annot_kws={"size": 10}
    fig.savefig(name + '.jpg')
    
def tocsv(label, cm):
    df = pd.DataFrame(cm)
    df.to_csv("pics/predict_ccsd_{}_{}.csv".format(label, sigma))

def run(snr):
    ccsd0 = []
    ccsd1 = []
    for i in mod:
        for j in snr:
            base = i*p + j*pp + 2
            s = x[base:(base + pts),:,0] + 1j*x[base:(base + pts),:,1]
            s = cyc.CCSD(s, 512, 512, sigma)
            s = block_reduce(s, block_size=(1, 1, 32), func=np.max).reshape((pts, -1))
            ccsd0.append(s[:tr, :])
            ccsd1.append(s[tr::, :])
            
    ccsd0 = np.asarray(ccsd0)
    ccsd1 = np.asarray(ccsd1)
    l = ccsd0.shape[-1]
    ccsd0 = ccsd0.reshape((-1, l))
    ccsd1 = ccsd1.reshape((-1, l))
    print (ccsd0.shape, ccsd1.shape)
    return ccsd0, ccsd1


#######################  HIGH SNR.
snr = range(18,26)
x_tr, x_te = run(snr)
yy_tr = create_label(len(mod), tr * len(snr))
yy_te = create_label(len(mod), te * len(snr))

fi.writelines('test ccsd image, high SNR. \n')
print ('start test')
sc, cm1 = fsr.lda(x_tr, yy_tr, x_te, yy_te)
fi.writelines('%f \n' % sc)

tocsv('high', cm1)

#######################  MID SNR.
snr = range(8,16)
x_tr, x_te = run(snr)
yy_tr = create_label(len(mod), tr * len(snr))
yy_te = create_label(len(mod), te * len(snr))

fi.writelines('test ccsd image, middle SNR. \n')
print ('start test')
sc, cm2 = fsr.lda(x_tr, yy_tr, x_te, yy_te)
fi.writelines('%f \n' % sc)

tocsv('middle', cm2)

#######################  LOW SNR.
snr = range(0,8)
x_tr, x_te = run(snr)
yy_tr = create_label(len(mod), tr * len(snr))
yy_te = create_label(len(mod), te * len(snr))

fi.writelines('test ccsd image, low SNR. \n')
print ('start test')
sc, cm3 = fsr.lda(x_tr, yy_tr, x_te, yy_te)
fi.writelines('%f \n' % sc)

tocsv('low', cm3)
'''
image('high', cm1)
image('middle', cm2)
image('low', cm3)
'''
