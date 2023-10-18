#!/usr/bin/env python

import numpy as np
import pandas as pd

from embedder import embedder
import torch
import exmol
import selfies as sf


# =================================================================================================


# note, this function processes one smiles at a time: whatever is given as the base
def get_colDen(smiles_str):
    # imports embedder and converts given smiles to selfies, then tokenizes into vectors
    embeddings = embedder.embed_smiles(smiles_str).numpy()

    # hard code parameters
    # t_rot, v_lsr, dv, isotopologue, environment code, excited state
    features = np.array([[100, 7.2, 10, 0, 1, 0]])

    # combines np.arrays to correct dim for ML model
    combined = np.hstack([features, embeddings])

    # predicts column density, returns 1D nd.array
    Ncol = gradient_boosting.predict(combined)

    return Ncol



def embedding_func(smiles_str: str):
    """
    Converts given smiles string into vector embedding as an numpy array
    :param smiles_str: starting SMILES : str

    :return: embedded vector : np.array
    """
    embeddings = embedder.embed_smiles(smiles_str).numpy().squeeze()

    return embeddings



def cos_similarity(a, b):
    """
    Computes the cosine similarity between the reference vector and counterfactual vector
    :param a: reference embedding vector : np.array
    :param b: cf embedding vector : np.array

    :return sim: calulcated cosine similarity
    """
    # calculate dot product
    ab_dot = np.dot(a, b.T)

    # calculate vector magnitudes
    a_mag = np.linalg.norm(a)
    b_mag = np.linalg.norm(b)

    # calculate cosine simularity
    cos_sim = ab_dot / (a_mag * b_mag)
    print(cos_sim)

    return cos_sim







