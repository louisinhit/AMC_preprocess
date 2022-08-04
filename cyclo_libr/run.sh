#! /bin/bash

#for i in 'logreg' 'lda' 'sgd' 
#do
#  python3 dataset_prepare.py --clf $i
#done

python3 scd.py
python3 cccsd.py
python3 chtc.py
python3 rdctcf.py
