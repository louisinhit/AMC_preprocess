import numpy as np
from matplotlib import cm
import h5py, pickle
import sys
sys.path.append('lib/')
import lib_interface as lib
import feature_selection_reduce as fsr

params = {  'window_size'    : 2048,
            'window_step'    : 1024,
            'dwt_wave'       : 'haar',
            'cwt_wave'       : 'gaus1',
            'cwt_scal'       : 4,
            'ccsd_sigma'     : 0.2,
            'ctcf_resolution': 1
            }

features = lib.Trans_Pool(cuda=False, save_to_file=False, **params)

####### chad
mod_list = ['bpsk', 'qpsk', '8psk', 'dqpsk', 'msk', '16qam', '64qam', '256qam']
dataset = np.load('../dataset/challenge_chad_dataset.dat')
leng = 2048
dataset = dataset[:, :, :leng].transpose((1, 0, 2))
print (dataset.shape)
out = features(dataset)
with open('CTCF_chi2_1024_chad_dataset.npy', 'rb') as f:
    s = np.load(f)
    
s = s.reshape((8, 2500, 4, 1024))
with open('CTCF_chi2_1024_chad_dataset_.npy', 'wb') as f:
    np.save(f, s) 
    
'''
s = []
with open('CTCF_2048_chad_dataset.npy', 'rb') as f:
    for i in range(dataset.shape[0]):
        s.append(np.load(f))

s = np.asarray(s).transpose((1, 0, 2, 3))
s = s.reshape((s.shape[0], -1, s.shape[-1]))

yy = []
for i in range(8):
    yy.append([i] * int(2500))
yy = np.hstack(yy)

out = []
for i in s:
    out.append(fsr.chi_2(i, yy, 1024))
    print ('one done..')

out = np.asarray(out).transpose((1, 0, 2))
print (out.shape)
with open('CTCF_chi2_1024_chad_dataset.npy', 'wb') as f:
    np.save(f, out)

with open('SCD_1024_chad_dataset.npy', 'wb') as f:
    for i in dataset:
        s = fsr.max_pool_2d(features.SCD_graph(i))
        np.save(f, np.float32(s))
        print (s.shape)


with open('CCSD_1024_chad_dataset.npy', 'wb') as f:
    for i in dataset:
        s = fsr.max_pool_2d(features.CCSD_graph(i))
        np.save(f, np.float32(s))
        print (s.shape)

########### RML
hf = h5py.File('../dataset/201801a_data_train.h5', 'r+')
x = hf['train']
p = int(len(x) / 24)
pp = int(p / 26)
data = []
print ('hi')
for i in range(26 * 24):
    xx = x[(i * pp):(i * pp + 64), :, 0] + 1j * x[(i * pp):(i * pp + 64), :, 1]
    data.append(np.float32(features.SCD_graph(xx)))
    print ('11111')
    exit()


with open('RML_scd_256_train.npy', 'wb') as f:
    np.save(f, np.asarray(data))

hf.close()

print ('start test.')

hf = h5py.File('../dataset/201801a_data_test.h5', 'r+')
x = hf['test']
p = int(len(x) / 24)
pp = int(p / 26)
data = []


for i in range(26 * 24):
    xx = x[(i * pp):(i * pp + pp), :, 0] + 1j * x[(i * pp):(i * pp + pp), :, 1]
    data.append(np.csingle(features.RD_CTCF(xx)))
    print (i)

data = np.asarray(data)
print (data.shape)

with open('RML_256_ctcf_test.npy', 'wb') as f:
    np.save(f, np.asarray(data))

hf.close()
'''

'''
####### perfect
hf = h5py.File('../generate_dataset/my_qam_SNR_5.h5', 'r+')
data = []
for k in hf.keys():
    data.append(np.float32(features.SCD_graph(hf[k][:, :])))

hf.close()
data = np.asarray(data)
print (data.shape)

with open('perfect_scd_160.npy', 'wb') as f:
    np.save(f, data)

print ('scd done.')

hf = h5py.File('../generate_dataset/my_qam_SNR_5.h5', 'r+')
data = []
for k in hf.keys():
    data.append(np.float32(features.CCSD_graph(hf[k][:, :])))

hf.close()
data = np.asarray(data)
print (data.shape)

with open('perfect_ccsd_160_0.5.npy', 'wb') as f:
    np.save(f, data)

'''

print ('all done.')

