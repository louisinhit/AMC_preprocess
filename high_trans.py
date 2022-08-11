import numpy as np
import h5py, sys, argparse
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression as LRG
from sklearn.linear_model import SGDClassifier as SGD
from sklearn.metrics import confusion_matrix
import seaborn as sn


parser = argparse.ArgumentParser()
parser.add_argument('--classifier', type=str, default='LDA')
parser.add_argument('--cuda', type=bool, default=True)
args = parser.parse_args()
classifier_ = args.classifier

if args.cuda:
    import ho_cupy as ho
else:
    import ho_numpy as ho


para = {'figure.figsize'  : (8, 6) }
plt.rcParams.update(para)

mod = range(19)
trans_list = ['element_HOS', 'RD_CTCF']
mod_list = ['OOK', '4ASK', '8ASK', 'BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK',\
         '32APSK', '64APSK', '128APSK', '16QAM', '32QAM', '64QAM', '128QAM', \
         '256QAM', 'GMSK', 'OQPSK']

hf = h5py.File('201801a_subset.h5', 'r+')
x_test = hf['test'][:, :, :]
x_train = hf['train'][:, :, :]
pts = 300
tr = int(pts // 10) * 9
te = pts - tr
hf.close()

fi = open("logbook_high_trans.txt", "w")
fi = open("logbook_high_trans.txt", "a")
fi.write("Start \n")

###############################################################
############################# train and test functions
###############################################################
def sys_out(msg):
    fi.writelines(msg)
    print (msg)


def create_label(num):
    mo = []
    for m in mod:
        mo.append([m] * num)
    mo = np.hstack(mo)
    return mo

def classifier(out_tr, yy_tr, out_te, yy_te):
    if classifier_ is 'LDA':
        lda = LDA().fit(out_tr, yy_tr)
        cm = confusion_matrix(yy_te, lda.predict(out_te))
        return lda.score(out_te, yy_te), cm

    elif classifier_ is 'SGD':
        clf = SGD(alpha=0.1, max_iter=100, shuffle=True, random_state=0, tol=1e-3)
        clf.fit(out_tr, yy_tr)
        cm = confusion_matrix(yy_te, clf.predict(out_te))
        return clf.score(out_te, yy_te), cm
    
    elif classifier_ is 'LRG':
        clf = LRG(random_state=0).fit(out_tr, yy_tr)
        cm = confusion_matrix(yy_te, clf.predict(out_te))
        return clf.score(out_te, yy_te), cm

    else:
        sys.exit(" WRONG CLASSIFIER NAME ! ")


def run(snr, trans):
    train = []
    test = []
    for i in mod:
        for j in snr:
            base = i * 26 + j
            s = x_train[base,:,:]
            s = getattr(ho, trans)(s)
            s = s.reshape((tr, -1))
            train.append(s)

            s = x_test[base,:,:]
            s = getattr(ho, trans)(s)
            s = s.reshape((te, -1))
            test.append(s)

    train = np.asarray(train)
    test = np.asarray(test)
    l = train.shape[-1]

    if np.iscomplexobj(train):
        train, test = np.abs(train), np.abs(test)
        
    train = train.reshape((-1, l))
    test = test.reshape((-1, l))
    print (train.shape, test.shape)
    return train, test


def train_test(T):

    snr = range(18,26)
    x_tr, x_te = run(snr, T)
    yy_tr = create_label(tr * len(snr))
    yy_te = create_label(te * len(snr))

    sys_out('start test high snr , transform is {}'.format(T))
    sc, cm = classifier(x_tr, yy_tr, x_te, yy_te)
    sys_out('the acc : %f' % sc)
    fig = plt.figure()
    name = "highSNR_{}".format(T)
    plt.title(name, fontsize = 10)
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(cm, xticklabels=mod_list, yticklabels=mod_list, cmap='Greens')
    fig.savefig("{}_confusion_matrix_high_SNR.png".format(T))


    snr = range(8,16)
    x_tr, x_te = run(snr, T)
    yy_tr = create_label(tr * len(snr))
    yy_te = create_label(te * len(snr))

    sys_out('start test middle snr , transform is {}'.format(T))
    sc, cm = classifier(x_tr, yy_tr, x_te, yy_te)
    sys_out('the acc : %f' % sc)
    fig = plt.figure()
    name = "middleSNR_{}".format(T)
    plt.title(name, fontsize = 10)
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(cm, xticklabels=mod_list, yticklabels=mod_list, cmap='Greens')
    fig.savefig("{}_confusion_matrix_middle_SNR.png".format(T))


    snr = range(0,8)
    x_tr, x_te = run(snr, T)
    yy_tr = create_label(tr * len(snr))
    yy_te = create_label(te * len(snr))

    sys_out('start test low snr , transform is {}'.format(T))
    sc, cm = classifier(x_tr, yy_tr, x_te, yy_te)
    sys_out('the acc : %f' % sc)
    fig = plt.figure()
    name = "lowSNR_{}".format(T)
    plt.title(name, fontsize = 10)
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(cm, xticklabels=mod_list, yticklabels=mod_list, cmap='Greens')
    fig.savefig("{}_confusion_matrix_low_SNR.png".format(T))


###############################################################
############################# main function
###############################################################
for tt in trans_list:
    sys_out("start {} train and test".format(tt))
    train_test(tt)

sys_out('DONE')
