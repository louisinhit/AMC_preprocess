import numpy as np
import h5py


mod = [i for i in range(0, 24) if i not in [17,18,19,20,21]]  # all digital mods.
snr = range(26)

hf = h5py.File('path/to/the/GOLD_XYZ_OSC.0001_1024.hdf5', 'r+')
x = hf['test']
p = int(len(x) / 24)
pp = int(p / 26)
pts = 300
tr = int(pts // 10) * 9

data_test = []
data_train = []
  
for i in mod:
    for j in snr:
        base = i*p + j*pp + 2
        s = x[base:(base + pts),:,0] + 1j*x[base:(base + pts),:,1]
        data_train.append(s[:tr, :])
        data_test.append(s[tr::, :])
        
data_train = np.asarray(data_train)
data_test = np.asarray(data_test)
print (data_train.shape)
print (data_test.shape)

# write to hdf5 file
hf = h5py.File('201801a_subset.h5', 'w')
hf.create_dataset('test', data=data_test)
hf.create_dataset('train', data=data_train)
hf.close()
