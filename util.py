import numpy as np
from sklearn.utils import shuffle

from variables import*
np.random.seed(seed)

def get_data():
    XsspA = np.random.randn(n_drug_pairs, n_ssp) + 1
    XtspA = np.random.randn(n_drug_pairs, n_tsp) + 1
    XgspA = np.random.randn(n_drug_pairs, n_gsp) + 1

    XsspB = np.random.randn(n_drug_pairs, n_ssp) + 1
    XtspB = np.random.randn(n_drug_pairs, n_tsp) + 1
    XgspB = np.random.randn(n_drug_pairs, n_gsp) + 1

    Y = np.random.choice([0, 9], size=(n_drug_pairs,))
    X = (XsspA, XtspA, XgspA, XsspB, XtspB, XgspB)

    return X, Y