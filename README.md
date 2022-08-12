# AMC_preprocess
## Python Signal Pre-processing Library for AMC (or more applications ... we certainly hope that this repository will not only be used for AMC ... hahaha)

This work corresponds to the IEEE Access paper: <https://ieeexplore.ieee.org/abstract/document/9852206>

If you want to use these code please cite our paper:
```
@ARTICLE{9852206,
  author={Liu, Xueyuan and Li, Carol Jingyi and Jin, Craig and Leong, Philip H.W.},
  journal={IEEE Access}, 
  title={Wireless Signal Representation Techniques for Automatic Modulation Classification}, 
  year={2022},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/ACCESS.2022.3197224}}
```
Initially, to reproduce the results, please download the dataset first: <https://www.kaggle.com/datasets/pinxau1000/radioml2018>

Our train and test only use a subset if it, as the original one has 20GB which is not good for LDA classifier. To generate the subset and then reproduce the experiments, just simply following:

1.	Change the `hf = h5py.File('path/to/the/GOLD_XYZ_OSC.0001_1024.hdf5', 'r+')` in `gen_subset.py` into the path that can navigate to the dataset.

2.	Use `train_test.sh` run the whole program, here you can choose different classifiers, if you have cuda please set `cuda=true` as it’ll be crazy slow if you only use `Numpy`.

3.	Obviously, this work has some dependency requirements, please use `pip` or `conda` install them first. (you could just follow the python errors to check what you need)

4.	This program will generate some `.txt` and `.png` automatically, including the test accuracy and confusion matrixes.

Please let me know if you have any question! You can email us or leave comments in Issues!

<maggieliuyuri@gmail.com>
