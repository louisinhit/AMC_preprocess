# AMC_preprocess
# Python Signal Pre-processing Library for AMC
Giving an input complex signal, and returning transformation outputs including: `['CCSD_graph', 'CCSD_profile', 'CWT', 'DWT', 'FT', 'HOS', 'RD_CTCF', 'SCD_graph', 'SCD_profile', 'STFT']`

Library is inside the `lib` folder.

- The current functions does not support pip operations and therefore requires to copy these files to 
where you need to implement transforms. For initial usage, you can start by playing with the `lib_usage_run_all_test.ipynb`.
- This library can be called by `import lib_interface as Name`
- Initialisation: All keywords arguments are passed to lib using dictionary datatype.

```
USE_CUDA = True                                     ### Do you have cuda? If you don't set to False.
SAVE = True                                         ### Save the multi-dimension output features to .h5 file?

input_data = # a numpy array with shape [n_classes, n_samples, signal_length] dtype = np.cfloat
length = input_data.shape[-1]                       ### The length of input signal, for example, 1024.

params = {  'window_size'    : int(length / 4),
            'window_step'    : int(length / 8),
            'dwt_wave'       : 'haar',
            'cwt_wave'       : 'gaus1',
            'cwt_scal'       : int(length / 8),
            'ccsd_sigma'     : 0.5,
            'ctcf_resolution': 2
            }                                       ### These are suggested settings, you can customize them.

# usage one: ONLY do transformations.
features = Name.Trans_Pool(cuda=USE_CUDA, save_to_file=SAVE, **params)
x1 = features(input_data.reshape((-1, length)))     ### Call the library and the return datatype is also dictionary.

# usage two: implement transformations and LDA classifier test.
# all test results including accuracy and running status are saved to a logbook named "Trans_and_test_logbook.txt"
features = Name.Trans_and_Test(cuda=USE_CUDA, save_to_file=SAVE, **params)
x2 = features(input_data)                           ### Call the library and the return datatype is also dictionary.
```

Some explanation of keywords arguments:
- `window_size` and `window_step`: Control the sliding window when the transformation involves in it. One-quarter and one-eighth of the input length are good enough.
- `dwt_wave`: Controls the wavelet type used in DWT, choose from `['bior1.1', 'bior1.3', 'bior1.5', 'bior2.2', 'bior2.4', 'bior2.6', 'bior2.8', 'bior3.1', 'bior3.3', 'bior3.5', 'bior3.7', 'bior3.9', 'bior4.4', 'bior5.5', 'bior6.8', 'coif1', 'coif2', 'coif3', 'coif4', 'coif5', 'coif6', 'coif7', 'coif8', 'coif9', 'coif10', 'coif11', 'coif12', 'coif13', 'coif14', 'coif15', 'coif16', 'coif17', 'db1', 'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8', 'db9', 'db10', 'db11', 'db12', 'db13', 'db14', 'db15', 'db16', 'db17', 'db18', 'db19', 'db20', 'db21', 'db22', 'db23', 'db24', 'db25', 'db26', 'db27', 'db28', 'db29', 'db30', 'db31', 'db32', 'db33', 'db34', 'db35', 'db36', 'db37', 'db38', 'dmey', 'haar', 'rbio1.1', 'rbio1.3', 'rbio1.5', 'rbio2.2', 'rbio2.4', 'rbio2.6', 'rbio2.8', 'rbio3.1', 'rbio3.3', 'rbio3.5', 'rbio3.7', 'rbio3.9', 'rbio4.4', 'rbio5.5', 'rbio6.8', 'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8', 'sym9', 'sym10', 'sym11', 'sym12', 'sym13', 'sym14', 'sym15', 'sym16', 'sym17', 'sym18', 'sym19', 'sym20']`
- `cwt_wave`: Controls the wavelet type used in CWT, choose from `['cgau1', 'cgau2', 'cgau3', 'cgau4', 'cgau5', 'cgau6', 'cgau7', 'cgau8', 'cmor', 'fbsp', 'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5', 'gaus6', 'gaus7', 'gaus8', 'mexh', 'morl', 'shan']`
- `cwt_scal`: The scale factor of CWT. One-eighth of the input length is good enough.
- `ccsd_sigma`: The sigma value in CCSD, common range is `[0.001, 1.0]`, 0.5 is good enough for RML dataset.
- `ctcf_resolution`: Determines the precision of `RD-CTCF`, 2 is good enough.

Please let me know if you have any question!
