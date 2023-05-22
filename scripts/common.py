#!/usr/bin/env python


from typing import Tuple
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import warnings

from joblib import load

from astrochem_embedding import VICGAE
import torch

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.utils import resample

#=================================================================================================

def smiles_check(xclass, embedder):
    '''
    This function checks to ensure each SMILES string is properly
    embedded. Any SMILES that are not accepted by the embedder are
    dropped from the df and returned as a list for review

    Parameters
    ----------
    xclass : DataFrame[pandas]
        df of features characterzing okl, only 'SMILES' and SMILES vector' 
        columns are considered here
    embedder : class[VICGAE model]
        takes in SMILES string and returns a 2D torch Tensor 
        specifically [np.ndarray[np.ndarray]]

    Returns
    -------
    SMILES strings that were not supported by the embedder 
    ''' 
    warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    removed = []
    # automatically incremented for loop for counter j instead of keeping the counter and increment yourself this will do it for you automatically
    for j, i in enumerate(xclass['SMILES']):
        try:
            xclass['SMILES vector'][j] = torch.squeeze(embedder.embed_smiles(i)).numpy()
            #print(xclass['SMILES vector'][j])
        except:
            removed.append(i)
            xclass.drop(labels=j, axis=0, inplace=True)

    #print(xclass['SMILES vector'])
    print('molecules not accepted by embedder:', removed)

    for i in range(len(xclass['SMILES vector'][0])): 
        xclass['column %s' % i] = xclass['SMILES vector'].str[i]

    xclass.drop('SMILES vector', axis=1, inplace=True)  

    return xclass, removed




def bootstrap_bill_turner(data: Tuple[np.ndarray], seed: int, n_samples: int = 500, replace: bool = True, noise_scale: float = 0.5, molecule_split: float = 0.2, test_size: float = 0.2):
    """
    This function specifically splits the training set into train
    and validation sets within molecule classes. The idea behind this
    is to prevent data leakage.

    Parameters
    ----------
    data : Tuple[np.ndarray]
        [description]
    seed : int
        [description]
    n_samples : int, optional
        [description], by default 500
    replace : bool, optional
        [description], by default True
    noise_scale : float, optional
        [description], by default 0.5
    molecule_split : float, optional
        [description], by default 0.2
    test_size : float
        [description], by default 0.2
    """
    true_X, true_y = data
    indices = np.arange(len(true_y))
    rng = np.random.default_rng(seed)
    # shuffle the molecules
    rng.shuffle(indices)
    split_num = int(len(indices) * molecule_split)
    test_indices = indices[:split_num]
    train_indices = indices[split_num:]
    # print('test', test_indices)
    # print('train', train_indices)
    test_indices.sort(); train_indices.sort()
    sets = list()
    indices = list()
    for index_set, train in zip([train_indices, test_indices], [True, False]):
        if train:
            num_samples = int(n_samples * (1 - test_size))
        else:
            num_samples = int(n_samples * test_size)
        resampled_indices = resample(index_set, n_samples=num_samples, replace=replace, random_state=seed)
        resampled_indices.sort()
        resampled_X, resampled_y = true_X[resampled_indices], true_y[resampled_indices]
        reshuffled_indices = np.arange(resampled_y.size)
        rng.shuffle(reshuffled_indices)
        resampled_y += rng.normal(0., noise_scale, size=resampled_y.size)
        sets.append(
            (resampled_X[reshuffled_indices], resampled_y[reshuffled_indices])
        )
        indices.append(resampled_indices[reshuffled_indices])
    return sets, np.concatenate(indices), test_indices, train_indices