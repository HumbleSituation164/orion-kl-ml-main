#!/usr/bin/env python

import numpy as np
import pandas as pd
from astrochem_embedding import VICGAE, get_paths, get_pretrained_path, Translator
import torch

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
    removed = []
    for j, i in enumerate(data["SMILES"]):
        try:
            data["SMILES vector"][j] = torch.squeeze(embedder.embed_smiles(i)).numpy()
        except:
            removed.append(i)
            data.drop(labels=j, axis=0, inplace=True)

    print("molecules not accepted by embedder:", removed)

    # seperates each embedding vector into indivdual columns
    for i in range(len(data["SMILES vector"][0])):
        data["column %s" % i] = data["SMILES vector"].str[i]
    data.drop("SMILES vector", axis=1, inplace=True)

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



