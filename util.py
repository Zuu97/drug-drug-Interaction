import numpy as np
from sklearn.utils import shuffle

from variables import*
np.random.seed(seed)

def get_data():
    Xssp = np.random.randn(n_drugs, n_ssp) + 1
    Xtsp = np.random.randn(n_drugs, n_tsp) + 1
    Xgsp = np.random.randn(n_drugs, n_gsp) + 1
    Y = np.random.choice([0, 1], size=(n_drugs,), p=[2./3, 1./3])
    X = (Xssp, Xtsp, Xgsp)

    return X, Y