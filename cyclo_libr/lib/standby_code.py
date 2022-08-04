import numpy as np
import random
import h5py, pickle, sys
import lib_interface as lib
import feature_selection_reduce as fsr

def scd_coef(s, l):
    N = s.shape[-1]
    an = torch.arange(0, 1, 1/N)
    if l < s.shape[-2] - 1:
        signal = torch.cat((s[:,l,:],s[:,l+1,:]), dim=-1)
    elif l == s.shape[-2] - 1:
        signal = torch.cat((s[:,1,:], torch.zeros((s.shape[0],N)).cuda()), dim=-1)
        #print("because of the range: append with zeros")
    else:
        sys.exit("the range of L is not correct")

    suanzi = torch.outer(-2j*torch.Tensor([math.pi])*an, torch.arange(0, N))
    suanzi = torch.exp(suanzi)  # alpha * win
    
    gm = torch.empty((N, signal.shape[0], N), dtype=torch.cfloat)
    for j in range(N):
        gm[j, :, :] = signal[:,0:N] * signal[:,j:(N+j)]
    
    # sample * tau * win
    return torch.einsum('mji,ki->mkj', torch.transpose(gm, 0, 1), suanzi) / N


def SCD(s, N, L):
    s = block_samples(s, N, L)
    r = []
    for l in range(s.shape[1]):
        r.append(scd_coef(s,l))
    r = torch.stack((r))
    r = torch.mean(torch.as_tensor(r, dtype=torch.cfloat), dim=0)
    return torch.abs(fftshift(fft(r, dim=-1), dim=(-2, -1))).cuda()


def correntropy_mean(s, l, sigma):
    N = s.shape[-1]
    if l< s.shape[-2]-1:
        signal = torch.cat((s[:,l,:],s[:,l+1,:]), dim=-1)
    elif l == s.shape[-2]-1:
        signal = torch.cat((s[:,1,:], torch.zeros((s.shape[0],N))), dim=-1)
    else:
        sys.exit("the range of L is not correct")

    gm = []
    for i in range(N):
        gm.append(signal[:,0:N]-signal[:,i:(N+i)])
        # shape N * sample * N
    gm = torch.asarray(gm, dtype=torch.cfloat).transpose((1, 0, 2))
    Gm = torch.exp(-torch.abs(gm)**2/(2*sigma**2)) / (torch.sqrt(2*torch.pi)*sigma)
    return Gm - torch.mean(Gm, dim=(-1,-2), keepdims=True)


def correntropy_coef(gm):
    N = gm.shape[-1]
    an = torch.arange(0, 1, 1/N)
    suanzi = torch.outer(-2j*torch.pi*an, torch.arange(0, N))
    suanzi = torch.exp(suanzi)  # alpha * win
    return torch.einsum('ki,mji->mkj', suanzi, gm) / N


def CCSD(s, N, L, sigma):
    s = block_samples(s, N, L)
    v = []
    for l in range(s.shape[1]):
        Gm = correntropy_mean(s, l, sigma)
        v.append(correntropy_coef(Gm))

    v = torch.mean(torch.asarray(v, dtype=torch.cfloat), dim=0)
    return torch.abs(fftshift(fft(v, dim=-1), dim=(-2, -1)))



'''
# the number of classes. trainset and test set size.
cls = 8
dataset_name = '../../dataset/challenge_chad_dataset.dat'


def create_label(num):
    mods = range(cls)
    mo = []
    for m in mods:
        mo.append([m] * num)
    mo = np.expand_dims(np.hstack(mo), axis=1)

    return mo

# import dataset, here use .dat format as example.
dataset = np.load(open(dataset_name, 'rb'))
# if use h5py
# x = h5py.File(dataset_name, 'r+')

# a subset of whole dataset for ranking. (8, 2500, 32768) complex64
subset = dataset[:, :40, :2048]
length = subset.shape[-1]

params = {  'window_size'    : int(length / 2),
            'window_step'    : int(length / 2),
            'dwt_wave'       : 'haar',
            'cwt_wave'       : 'gaus1',
            'cwt_scal'       : 2,
            'ccsd_sigma'     : 0.3,
            'ctcf_resolution': 1
            }
# Do you have cuda? Want to save output features to file?
USE_CUDA = True
# usage one: only do transformations.
features = lib.LDA_rank(cuda=USE_CUDA, save_to_file=False, best_no=4, **params)
best, _ = features.rank(subset)

print (best)


#######################
#best = ['CCSD_profile', 'RD_CTCF', 'STFT', 'SCD_profile', 'FT', 'DWT']
dataset = dataset[:, :, :2048]
_, pis, length = dataset.shape
new_length = int(length / 4)

params = {  'window_size'    : new_length,
            'window_step'    : new_length,
            'dwt_wave'       : 'haar',
            'cwt_wave'       : 'gaus1',
            'cwt_scal'       : 2,
            'ccsd_sigma'     : 0.3,
            'ctcf_resolution': 1
            }
'''
#features = lib.Preprocess_Transform(cuda=True, save_to_file=False, **params)
#features.transfer_and_compress(dataset, best)
'''

features = lib.Trans_Pool(cuda=True, save_to_file=False, **params)
# for large dataset, i suggest use np.save that saves to file one by one, saving time.
pp = 40
labels = create_label(pp)

for name in best:

    with open('{}_transfer_dataset_{}.npy'.format(name, new_length), 'wb') as f:
        for i in range(0, pis, pp):

            out = getattr(features, name)(dataset[:,i:(i+pp),:].reshape((-1, length)))
            if np.iscomplexobj(out):
                out = np.float32(abs(out))

            if name in ['CCSD_profile', 'CWT', 'DWT', 'FT', 'RD_CTCF', 'SCD_profile', 'STFT', 'Raw_IQ', 'CCSD_graph', 'SCD_graph']:
                out = fsr.chi_2(out.reshape((cls*pp, -1)), labels, new_length)

            elif name in ['HOCs']:
                t = int(new_length // 14)
                r = int(new_length % 14)
                out = np.concatenate((np.tile(out, t), out[:, :r]), axis=1)
            else:
                sys.exit("wrong transform key!")
            
            print (out.shape)
            np.save(f, out)
    print (name, 'is done..')
    
print ('all transfer and dimension reduce is done..')
'''
cls = 8
best = ['HOCs', 'CWT', 'DWT', 'CCSD_graph']
pis = 1201
pp = 40
new_length = 512 
transfer_dataset = []
for name in best:
    with open('{}_transfer_dataset_{}.npy'.format(name, new_length), 'rb') as f:
        for i in range(0, pis, pp):
            print (np.load(f).shape, i)
            transfer_dataset.append(np.load(f, allow_pickle=True))
    print (name, 'done')
transfer_dataset = np.asarray(transfer_dataset).reshape((len(best), -1, cls, pp, new_length))

transfer_dataset = transfer_dataset.transpose((2, 1, 3, 0, 4)).reshape((cls, -1, len(best), new_length))

print ('save to file: ', transfer_dataset.shape)
with open('Best_{}_transfer_dataset_{}.npy'.format(len(best), new_length), 'wb') as f:
    np.save(f, transfer_dataset)

    def trans_rank(self, n_signal):
        save_filename = "Trans_before_test_logbook_clf_{}.txt".format(self.classifier)
        fi = open(save_filename, "w")
        fi.write("Start! \n")
        fi = open(save_filename, "a")
        
        # n_signal shape : (classes, samples, length) in np.cfloat
        n_class, n_sample, l = n_signal.shape
        
        M = [x for x in dir(Trans_Pool) if not x.startswith('__')]
        output_s('***The following transformations will be implemented automatically*** \n', save_filename)
        output_s(','.join(M), save_filename)
        
        best = {}
        tr = int(n_sample // 8) * 7
        
        yy_tr = create_label(n_class, tr)
        yy_te = create_label(n_class, int(n_sample-tr))

        output_s('lable shape:{} {}'.format(yy_tr.shape, yy_te.shape), save_filename)
        for uu in M:
            output_s('Currently Running:' + uu, save_filename)
            out_tr = []
            out_te = []
            for c in range(n_class):
                out_tr.append(getattr(self, uu)(n_signal[c,:tr,:]))
                out_te.append(getattr(self, uu)(n_signal[c,tr::,:]))
                output_s('--- Class %d is done!' % c, save_filename)
            
            out_tr = np.asarray(out_tr).reshape((n_class * tr, -1))
            out_te = np.asarray(out_te).reshape((n_class * (n_sample-tr), -1))

            if uu in ['HOCs']:
                t = int(self.window_size * 2 // 14)
                r = int(self.window_size * 2 % 14)
                out_tr = np.concatenate((np.tile(out_tr, t), out_tr[:, :r]), axis=1)
                out_te = np.concatenate((np.tile(out_te, t), out_te[:, :r]), axis=1)
            '''
            if np.iscomplexobj(out_tr):
                out_tr = fsr.chi_2(abs(out_tr), yy_tr, self.window_size)
                out_te = fsr.chi_2(abs(out_te), yy_te, self.window_size)
            else:
            '''
            out_tr = fsr.chi_2(abs(out_tr), yy_tr, self.window_size)
            out_te = fsr.chi_2(abs(out_te), yy_te, self.window_size)

            sc = getattr(fsr, self.classifier)(out_tr, yy_tr, out_te, yy_te)
            best[uu] = sc
            output_s('Accuracy for {} is {}'.format(uu, sc), save_filename)
        
        output_s('done.', save_filename)
        fi.close()

        best = sorted(best.items(), key=lambda x: x[1], reverse=True)  #[:self.best_no]
        best = [i[0] for i in best]
        output_s('top {} best features are: {}'.format(self.best_no, best), save_filename)

        return best


class Feature_Compress(Trans_Pool):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        
    def transfer_only(self, dataset, best_list):
        if type(dataset).__module__ != np.__name__ or len(dataset.shape) != 2:
            sys.exit("the dataset should be 2D numpy array")
        output = {}
        for uu in best_list:
            output[uu] = getattr(self, uu)(dataset)
        return output

    def reduce_only(self, dataset, labels):
        # dataset {M}[n_samples * classes, length]  labels [n_samples * classes]
        # 'CCSD_graph', 'CCSD_profile', 'CWT', 'DWT', 'FT', 'HOCs', 'RD_CTCF', 'SCD_graph', 'SCD_profile', 'STFT'
        
        out = []
        for k in dataset.keys():
            n = dataset[k].shape[0]

            if k in ['CCSD_profile', 'CWT', 'DWT', 'FT', 'RD_CTCF', 'SCD_profile', 'STFT', 'Raw_IQ', 'CCSD_graph', 'SCD_graph']:
                out.append(fsr.chi_2(dataset[k].reshape((n, -1)), labels, self.window_size * 2))

            elif k in ['HOCs']:
                t = int(self.window_size * 2 // 14)
                r = int(self.window_size * 2 % 14)
                out.append(np.concatenate((np.tile(dataset[k], t), dataset[k][:, :r]), axis=1))
            else:
                sys.exit("wrong transform key!")

        out = np.asarray(out).transpose((1, 0, 2))
        return out
        
    def transfer_and_compress(self, dataset, best):

        cls, pis, length = dataset.shape

        # for large dataset, i suggest use np.save that saves to file one by one, saving time.
        pp = 50
        labels = create_label(cls, pp)

        for name in best:

            with open('{}_transfer_dataset_{}.npy'.format(name, length), 'wb') as f:
                for i in range(0, pis, pp):

                    out = getattr(self, name)(dataset[:,i:(i+pp),:].reshape((-1, length)))
                    if np.iscomplexobj(out):
                        out = np.float32(abs(out))

                    if name in ['CCSD_profile', 'CWT', 'DWT', 'FT', 'RD_CTCF', 'SCD_profile', 'STFT', 'Raw_IQ', 'CCSD_graph', 'SCD_graph']:
                        # you can diy, using other functions in fsr.
                        out = fsr.chi_2(out.reshape((cls*pp, -1)), labels, length)

                    elif name in ['HOCs']:
                        t = int(length // 14)
                        r = int(length % 14)
                        out = np.concatenate((np.tile(out, t), out[:, :r]), axis=1)
                    else:
                        sys.exit("wrong transform key!")
            
                    print (out.shape)
                    np.save(f, out)
            print (name, 'is done..')
    
        print ('all transfer and dimension reduce is done..')
        del dataset
        transfer_dataset = []
        for name in best:
            with open('{}_transfer_dataset_{}.npy'.format(name, length), 'rb') as f:
                for i in range(0, pis, pp):
                    transfer_dataset.append(np.load(f))

        transfer_dataset = np.asarray(transfer_dataset).reshape((len(best), -1, cls, pp, length))
        transfer_dataset = transfer_dataset.transpose((2, 1, 3, 0, 4)).reshape((cls, -1, len(best), length))

        print ('save to file: ', transfer_dataset.shape)

        with open('Best_{}_transfer_dataset_{}.npy'.format(len(best), length), 'wb') as f:
            np.save(f, transfer_dataset)

