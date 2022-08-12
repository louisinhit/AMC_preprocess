#! /bin/bash

classifier="LDA"   # choose from 'LRG' 'LDA' 'SGD' 
cuda=true   # otherwise "False"

#python3 gen_subset.py
python3 linear_trans.py
python3 second_trans.py  --classifier $classifier  --cuda $cuda
python3 high_trans.py  --classifier $classifier  --cuda $cuda
