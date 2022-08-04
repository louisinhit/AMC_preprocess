import numpy as np
import random
import h5py, pickle, sys
sys.path.append('lib/')
import lib_interface as lib
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--clf', type=str, default='logreg')
parser.add_argument('--ws', type=int, default=128)
args = parser.parse_args()
clf = args.clf
ws = args.ws

'''
## for chad.
dataset_name = '../dataset/challenge_chad_dataset.dat'
dataset = np.load(open(dataset_name, 'rb'))
# a subset of whole dataset for ranking. (8, 2500, 32768) complex64
subset = dataset[:, :320, :1216]
length = subset.shape[-1]
'''
hf = h5py.File('../generate_dataset/my_qam_SNR_inf.h5', 'r+')
dataset = []
for k in hf.keys():
    print (k)
    dataset.append(hf[k][:, :])

hf.close()
dataset = np.asarray(dataset)
print (dataset.shape, dataset.dtype)

'''
## for RML.
hf = h5py.File('../dataset/201801a_data_test.h5', 'r+')
x = hf['test']
p = int(len(x) / 24)
pp = int(p / 26)
snr = range(24, 25)
mod_list = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK',\
         '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', \
         '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']
mod = [i for i in range(0, 24) if i not in [17,18,19,20,21]]

dataset = []
for i in mod:
    for j in snr:
        base = i*p + j*pp
        dataset.append(x[base:(base+pp),:,0] + 1j*x[base:(base+pp),:,1])

dataset = np.asarray(dataset).reshape((len(mod), -1, 1024))
print (dataset.shape, dataset.dtype)
hf.close()
del x
'''

params = {  'window_size'    : ws,
            'window_step'    : ws,
            'dwt_wave'       : 'haar',
            'cwt_wave'       : 'gaus1',
            'cwt_scal'       : 2,
            'ccsd_sigma'     : 0.2,
            'ctcf_resolution': 1
            }

features = lib.Feature_Rank(cuda=False, save_to_file=False, best_no=11, classifier=clf, **params)
best, _ = features.rank(dataset, ['SCD_graph', 'SCD_profile'])

print (best)

