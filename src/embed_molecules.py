#!/usr/bin/env python

from typing import Tuple
import numpy as np
import pandas as pd
from astrochem_embedding import VICGAE, get_paths, get_pretrained_path, Translator
import torch

from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.utils import resample

import selfies as sf

# ==========================================================================

embedder = VICGAE.from_pretrained()


def check_smiles(data: pd.DataFrame, embedder: torch.nn.Module):
    """
    This function checks to ensure each SMILES string is properly translated into a SELFIES string
    for embedding. Any SMILES that are not accepted by the embedder are
    dropped from the df and returned as a list for review.

    Parameters
    ----------
    data : DataFrame[pandas]
        df of features characterzing okl, only 'SMILES' and 'SMILES vector'
        columns are considered here
    embedder : class[VICGAE model]
        Accepts SMILES string and returns a 2D torch Tensor
        specifically [np.ndarray[np.ndarray]]

    Returns
    -------
    Dataframe appended with embedding vectors

    List of SMILES strings not supported by embedder

    """
    data['SMILES vector'] = ''
    removed = []
    for j, i in enumerate(data['SMILES']):
        try:
            data['SMILES vector'][j] = torch.squeeze(embedder.embed_smiles(i)).numpy()
        except:
            removed.append(i)
            data.drop(labels=j, axis=0, inplace=True)

    print("molecules not accepted by embedder:", removed)

    # seperates each embedding vector into indivdual columns
    for i in range(len(data['SMILES vector'][0])):
        data["column %s" % i] = data['SMILES vector'].str[i]
    data.drop('SMILES vector', axis=1, inplace=True)

    return data, removed


def map_to_embeddings(selfie: str):
    """
    This function tokenizes and then embeds a given SELFIE string
    This function can be used in a loop and will map each SELFIE to the corresponding embedding matrix

    Note: you will need to load the embedder where you call this function, otherwise loading the
    embedder within this function becomes costly

    Parameters
    ----------
    selfie : str

    Returns
    -------
    embedding : torch.Tensor
    	Molecular embeddings

    """
    vocab = Translator.from_yaml(get_pretrained_path().joinpath("translator.yml"))
    labels, tokens = vocab.tokenize_selfies(selfie)

    embedding = embedder.embed_molecule(torch.LongTensor(labels)).squeeze()
    return embedding



def bootstrap_bill_turner(data: Tuple[np.ndarray], seed: int, n_samples: int = 500, replace: bool = True, noise_scale: float = 0.5, molecule_split: float = 0.2, test_size: float = 0.2):
    """
    This function specifically splits the training set into train
    and validation sets within molecule classes. The idea behind this
    is to prevent data leakage.

    Parameters
    ----------
    data : Tuple[np.ndarray]
        Input data to be split into training/testing sets
    seed : int
        seed for random sampling in train/test split
    n_samples : int, optional
        number of samples to generate, by default 500
    replace : bool, optional
        sets random sampling with replacement, by default True
    noise_scale : float, optional
        sets distribution of simulated noise, by default 0.5
    molecule_split : float, optional
        [description], by default 0.2
    test_size : float
    	portion of dataset reserved for validation, by default 0.2

    Returns
    -------
    Training and testing sets with indicies

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

