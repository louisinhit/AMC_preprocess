from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from skimage.measure import block_reduce
from sklearn.decomposition import PCA
import sys
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix

#  clf for ranking.

def lda(out_tr, yy_tr, out_te, yy_te):
    lda = LDA().fit(out_tr, yy_tr)
    cm = confusion_matrix(yy_te, lda.predict(out_te))
    return lda.score(out_te, yy_te), cm


def sgd(out_tr, yy_tr, out_te, yy_te):
    clf = SGDClassifier(alpha=0.1, max_iter=100, shuffle=True, random_state=0, tol=1e-3)
    clf.fit(out_tr, yy_tr)
    cm = confusion_matrix(yy_te, clf.predict(out_te))
    return clf.score(out_te, yy_te), cm


def logreg(out_tr, yy_tr, out_te, yy_te):
    clf = LogisticRegression(random_state=0).fit(out_tr, yy_tr)
    cm = confusion_matrix(yy_te, clf.predict(out_te))
    return clf.score(out_te, yy_te), cm


# supervised feature compression.
def chi_2(x, y, k):
    """
    apply chi2 to select k best.
    data: x (n_samples, len); label: y (n_samples); k best: int
    return shape: (n_samples, new_len)
    """
    return SelectKBest(chi2, k=k).fit_transform(x, y)
    

def rfe(x, y, k):
    """
    return (n_samples, n_features_new)
    """
    return RFE(estimator=LogisticRegression(), n_features_to_select=k).fit_transform(x, y)


# unsupervised
def pca(x, k):
    return PCA(n_components=k).fit_transform(x)


def max_pool_2d(x, k=4):
    if len(x.shape) != 3:
        sys.exit("should be 2d image!")
    return block_reduce(x, block_size=(1, k, k), func=np.max)
    
    