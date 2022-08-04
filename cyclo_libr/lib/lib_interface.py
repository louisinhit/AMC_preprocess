import numpy as np
import pywt
from scipy.fft import fft, fftshift
import scipy.signal as scp
import h5py, pickle, sys, os
import feature_selection_reduce as fsr
import pandas as pd


class Trans_Pool:
    def __init__(self, cuda=False, save_to_file=False, **kwargs):
        self.save = save_to_file
        self.__dict__.update(kwargs)
        if cuda:
            self.so = __import__('second_order_cupy')
            self.ho = __import__('high_order_cupy')
        else:
            self.so = __import__('second_order_numpy')
            self.ho = __import__('high_order_numpy')
    
    def Raw_IQ(self, signal):
        return np.stack([signal.real, signal.imag], axis=1)
    
    def FT(self, signal, axis=-1):
        return fftshift(fft(signal, axis=axis))
    
    def STFT(self, signal):
        _, __, data = scp.stft(signal, nperseg=self.window_size, \
                               noverlap=int(self.window_size / 2), return_onesided=False)
        return data
    
    def DWT(self, signal):
        aa, dd = pywt.dwt(signal, self.dwt_wave, axis=-1)
        return np.concatenate((aa, dd), axis=-1)

    def CWT(self, signal):
        Fc = pywt.central_frequency(self.cwt_wave)
        Fc = 2 * Fc * self.cwt_scal
        scal = Fc / np.arange(1, self.cwt_scal + 1)
        coef, _ = pywt.cwt(signal, scal, self.cwt_wave, axis=-1)
        return coef.transpose((1, 0, 2))
    
    def SCD_graph(self, signal):
        return self.so.SCD(signal, self.window_size, self.window_step, algorithm=self.scd_alg)
    
    def SCD_profile(self, signal):
        signal = self.so.SCD(signal, self.window_size, self.window_step, algorithm=self.scd_alg)
        return np.concatenate((np.amax(signal, axis=-1), np.amax(signal, axis=-2)), axis=-1)
    
    def CCSD_graph(self, signal):
        return self.so.CCSD(signal, self.window_size, self.window_step, self.ccsd_sigma)
    
    def CCSD_profile(self, signal):
        signal = self.CCSD_graph(signal)
        return np.concatenate((np.amax(signal, axis=-1), np.amax(signal, axis=-2)), axis=-1)

    def HOCs(self, signal):
        return self.ho.HOC_element(signal)
    
    def RD_CTCF(self, signal):
        H = self.ho.HOC_cyclic(window_size=self.window_size, step=self.window_step, res=self.ctcf_resolution)
        return H(signal)
    
    def __call__(self, signal):
        out = {}
        M = [x for x in dir(Trans_Pool) if not x.startswith('__')]
        print('***The following transformations will be calculated automatically*** \n', M)
        for uu in M:
            out[uu] = getattr(self, uu)(signal)
            print(uu + ' is done!')
        print('***The return datatype is dictionary!***')
        if self.save:
            hf = h5py.File('Full_Feature_Transform_Results.h5', 'w')
            for i in (out.keys()):
                print (i, out[i].shape)
                hf.create_dataset(i, data=out[i])
            hf.close()
        return out


def output_s(message, save_filename):
    print (message)
    with open(save_filename, 'a') as out:
        out.writelines(message)
        out.writelines('\n')


def create_label(cls, num):
    mods = range(cls)
    mo = []
    for m in mods:
        mo.append([m] * num)
    mo = np.hstack(mo)
    return mo


class Feature_Rank(Trans_Pool):
    def __init__(self, save_to_file=False, classifier='lda', best_no=11, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.save = save_to_file
        self.best_no = best_no
        self.classifier = classifier

    def rank(self, n_signal, M=None):
        path = os.path.dirname(os.path.abspath("__file__"))
        save_filename = path + "/Trans_and_test_logbook_clf_{}_{}.txt".format(self.classifier, self.window_size)
        fi = open(save_filename, "w")
        fi.write("Start! \n")
        fi = open(save_filename, "a")
        
        # n_signal shape : (classes, samples, length) in np.cfloat
        n_class, n_sample, l = n_signal.shape
        
        if M is None:
            M = [x for x in dir(Trans_Pool) if not x.startswith('__')]
        
        output_s('***The following transformations will be implemented automatically*** \n', save_filename)
        output_s(','.join(M), save_filename)
        
        output = {}
        best = {}
        tr = int(n_sample // 10) * 9
        
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

            output[uu] = np.concatenate([out_tr.reshape((n_class, tr, -1)), out_te.reshape((n_class, (n_sample-tr), -1))], axis=1)
            
            if np.iscomplexobj(out_tr):
                output_s('-- ABS is applied bcs complex', save_filename)
                sc, cm = getattr(fsr, self.classifier)(abs(out_tr), yy_tr, abs(out_te), yy_te)
            else:
                sc, cm = getattr(fsr, self.classifier)(out_tr, yy_tr, out_te, yy_te)
            
            best[uu] = sc
            df = pd.DataFrame(cm)
            df.to_csv(path + "/predict_{}_{}_{}.csv".format(uu, self.classifier, self.window_size))

            output_s('Accuracy for {} is {}'.format(uu, sc), save_filename)
        
        output_s('done.', save_filename)
        fi.close()
        if self.save:
            hf = h5py.File('Full_Feature_Transform_Results.h5', 'w')
            for i in (output.keys()):
                print (i, output[i].shape)
                hf.create_dataset(i, data=output[i])
            hf.close()
        
        best = sorted(best.items(), key=lambda x: x[1], reverse=True)  #[:self.best_no]
        best = [i[0] for i in best]
        output_s('top {} best features are: {}'.format(self.best_no, best), save_filename)
        '''
        for key in M:
            if key not in best:
                del output[key]
        '''
        return best, output
    
    