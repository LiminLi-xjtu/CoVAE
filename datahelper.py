import sys, re, math, time
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
import collections
from collections import OrderedDict
import pandas as pd
from matplotlib.pyplot import cm



CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21,
               "V": 22, "Y": 23, "X": 24,
               "Z": 25}

CHARPROTLEN = 25

CHARCANSMISET = {"#": 1, "%": 2, ")": 3, "(": 4, "+": 5, "-": 6,
                 ".": 7, "1": 8, "0": 9, "3": 10, "2": 11, "5": 12,
                 "4": 13, "7": 14, "6": 15, "9": 16, "8": 17, "=": 18,
                 "A": 19, "C": 20, "B": 21, "E": 22, "D": 23, "G": 24,
                 "F": 25, "I": 26, "H": 27, "K": 28, "M": 29, "L": 30,
                 "O": 31, "N": 32, "P": 33, "S": 34, "R": 35, "U": 36,
                 "T": 37, "W": 38, "V": 39, "Y": 40, "[": 41, "Z": 42,
                 "]": 43, "_": 44, "a": 45, "c": 46, "b": 47, "e": 48,
                 "d": 49, "g": 50, "f": 51, "i": 52, "h": 53, "m": 54,
                 "l": 55, "o": 56, "n": 57, "s": 58, "r": 59, "u": 60,
                 "t": 61, "y": 62}

CHARCANSMILEN = 62

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64



def orderdict_list(dict):
    x = []
    for d in dict.keys():
        x.append(dict[d])
    return x

def one_hot_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros((MAX_SMI_LEN, len(smi_ch_ind)))  # +1

    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i, (smi_ch_ind[ch] - 1)] = 1

    return X  


def one_hot_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros((MAX_SEQ_LEN, len(smi_ch_ind)))
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i, (smi_ch_ind[ch]) - 1] = 1

    return X  


def label_smiles(line, MAX_SMI_LEN, smi_ch_ind):
    X = np.zeros(MAX_SMI_LEN)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smi_ch_ind, y
        X[i] = smi_ch_ind[ch]

    return X  


def label_sequence(line, MAX_SEQ_LEN, smi_ch_ind):
    X = np.zeros(MAX_SEQ_LEN)

    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]

    return X  


def get_removelist(list_name, length):
    removelist = []
    # Davis  SMILES:85 protein:1200   KIBA    SMILES:100   protein:1000
    for i, x in enumerate(list_name):
        if len(x) >= length:
            removelist.append(i)
    return removelist


def list_remove(list_name, removelist):
    a_index = [i for i in range(len(list_name))]
    a_index = set(a_index)
    b_index = set(removelist)
    index = list(a_index - b_index)
    a = [list_name[i] for i in index]
    return a


def df_remove(dataframe, removelist, axis):
    if axis == 0:
        new_df = dataframe.drop(removelist)
        new_df = new_df.reset_index(drop=True)
    if axis == 1:
        new_df = dataframe.drop(removelist, axis=1)
        new_df.columns = range(new_df.shape[1])
    return new_df


## ######################## ##
#
#  DATASET Class
#
## ######################## ## 
# works for large dataset
class DataSet(object):
    def __init__(self, fpath, setting_no, seqlen, smilen,need_shuffle=False):

        self.SEQLEN = seqlen
        self.SMILEN = smilen
        self.charseqset = CHARPROTSET
        self.charseqset_size = CHARPROTLEN

        self.charsmiset = CHARISOSMISET  ###HERE CAN BE EDITED
        self.charsmiset_size = CHARISOSMILEN
        self.PROBLEMSET = setting_no


    def parse_data(self, FLAGS, with_label=True):
        

        data_path = FLAGS.dataset_path
        ligands = json.load(open(data_path + "ligands_iso.txt"), object_pairs_hook=OrderedDict)
        proteins = json.load(open(data_path + "proteins.txt"), object_pairs_hook=OrderedDict)
        ligands = orderdict_list(ligands)
        proteins = orderdict_list(proteins)
        if 'davis' in data_path:
            affinities = pd.read_csv(data_path + 'drug-target_interaction_affinities_Kd__Davis_et_al.2011v1.txt',
                                     sep='\s+',
                                     header=None, encoding='latin1')  ### TODO: read from raw
            affinities = -(np.log10(affinities / (math.pow(10, 9))))
        else:
            affinities = pd.read_csv(data_path + 'kiba_binding_affinity_v2.txt', sep='\s+', header=None,
                                     encoding='latin1')
            ligands_remove = get_removelist(ligands, 90)
            proteins_remove = get_removelist(proteins, 1365)
            ligands = list_remove(ligands, ligands_remove)
            proteins = list_remove(proteins, proteins_remove)
            affinities = df_remove(affinities, ligands_remove, 0)
            affinities = df_remove(affinities, proteins_remove, 1)

        XD = []
        XT = []
        if with_label:
            for d in ligands:
                XD.append(label_smiles(d, self.SMILEN, self.charsmiset))

            for t in proteins:
                XT.append(label_sequence(t, self.SEQLEN, self.charseqset))
        else:
            for d in ligands.keys():
                XD.append(one_hot_smiles(ligands[d], self.SMILEN, self.charsmiset))

            for t in proteins.keys():
                XT.append(one_hot_sequence(proteins[t], self.SEQLEN, self.charseqset))

        return XD, XT, np.array(affinities)
