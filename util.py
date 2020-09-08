import numpy as np
from sklearn.utils import shuffle
import pandas as pd
from variables import*
np.random.seed(seed)

def generate_drug_pairs():
    drug_pairs = []
    drug_ids = np.arange(n_drugs)
    for i in range(n_drug_pairs):
        drug_pair = np.random.choice(drug_ids, 2, replace=False).tolist()
        if (drug_pair not in drug_pairs):
            for pair_ in drug_pairs:
                if (drug_pair == pair_[::-1]):
                    break
            else:
                drug_pairs.append(drug_pair)
    return drug_pairs

def drug_data():

    Xssp = np.random.randn(n_drugs, n_ssp) + 1
    Xtsp = 0.5 * np.random.randn(n_drugs, n_tsp) + 2
    Xgsp = 2 * np.random.randn(n_drugs, n_gsp) + 0.5
    return Xssp, Xtsp, Xgsp

def get_data():
    drug_pairs = generate_drug_pairs()
    Xssp, Xtsp, Xgsp = drug_data()

    A_drugs = [drug_pair[0] for drug_pair in drug_pairs]
    B_drugs = [drug_pair[1] for drug_pair in drug_pairs]

    XsspA = np.array([Xssp[i] for i in A_drugs])
    XtspA = np.array([Xtsp[i] for i in A_drugs])
    XgspA = np.array([Xgsp[i] for i in A_drugs])

    XsspB = np.array([Xssp[i] for i in B_drugs])
    XtspB = np.array([Xtsp[i] for i in B_drugs])
    XgspB = np.array([Xgsp[i] for i in B_drugs])

    XsspPair = np.concatenate((XsspA, XsspB), axis=1)
    XtspPair = np.concatenate((XtspA, XtspB), axis=1)
    XgspPair = np.concatenate((XgspA, XgspB), axis=1)
    Y = np.random.choice(np.arange(dense4), size=(len(drug_pairs),))
    return XsspPair, XtspPair, XgspPair, Y

def get_prediction_data(A_drug, B_drug):
    Xssp, Xtsp, Xgsp = drug_data()
    # print(" Enter drug id less than {}".format(n_drugs))
    # A_drug = int(input(" Enter first  drug id:"))
    # B_drug = int(input(" Enter second drug id:"))
    A_drug, B_drug = int(A_drug), int(B_drug)
    Xssp = np.concatenate((Xssp[A_drug], Xssp[B_drug]))
    Xtsp = np.concatenate((Xtsp[A_drug], Xtsp[B_drug]))
    Xgsp = np.concatenate((Xgsp[A_drug], Xgsp[B_drug]))
    return [[Xssp], [Xtsp], [Xgsp]]