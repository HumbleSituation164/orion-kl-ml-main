#!/usr/bin/env python

import numpy as np
import pandas as pd

from astrochem_embedding import VICGAE
import torch
from joblib import load, dump

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from embed_molecules import check_smiles

#========================================================


def make_sim_parameters(
    rot_temp: Tuple[float],
    vlsr: Tuple[float],
    dv: Tuple[float],
    isotopologue: int,
    env_code: int,
    excited_state: int,
    num_samples: int,
    embedding_vector: np.ndarray,
    seed: int = 1201354,
):
    """
    Generates a numpy array for concatenating with embeddings.
    This will draw random samples from specificied mean/variance
    of Gaussian distributions, and track them on the embedding
    vectors ready for regressor consumption.

    Parameters
    ----------
    rot_temp : Tuple[float]
        Tuple of rotational temperature value and error
    vlsr : Tuple[float]
        Tuple of vlsr value and error
    dv : Tuple[float]
        Tuple of dV value and error
    isotopologue : int
        binary flag for isotopologues 
        0 == False, 1 == True
    env_code : int
        integer input to indicate orion-kl environment of detection and prediction,
        0 == hot core, 1 == hot core s, 2 == compact ridge, 3 == plateau, 4 == extended ridge 
    excited_state : int
        binary flag for excited state molecules
        0 == False, 1 == True
    num_samples : int
        number of samples to generate
    embedding_vector: ndarray[numpy]
        numpy array of embedding features 
    seed : int
        integer value for seed, by default 1201354

    Returns
    -------
        Stacked numpy array of physical features and embedding features


    """
    rng = np.random.default_rng(seed)
    # * fancy syntax to unpack the tuple so easier way to write rot_temp[0], rot_temp[1], rot_temp[2]...
    temperatures = rng.normal(*rot_temp, num_samples)
    velocities = rng.normal(*vlsr, num_samples)
    linewidths = rng.normal(*dv, num_samples)
    iso = np.array([isotopologue] * num_samples)
    environment = np.array([env_code] * num_samples)
    excited_state = np.array([excited_state] * num_samples)

    combined = np.vstack(
        [temperatures, velocities, linewidths, iso, environment, excited_state]
    ).T

    # combined is a [num_samples, 6] shape array; we want to concat wth the embedding vectors
    embedding_vectors = embedding_vector[np.newaxis, :].repeat(num_samples, axis=0)
    # print(combined.shape, embedding_vectors.shape)

    return np.hstack([combined, embedding_vectors])






