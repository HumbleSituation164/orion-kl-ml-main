import torch
from torch import nn
from typing import Union, Iterable

import numpy as np
from astrochem_embedding import VICGAE


def embed_smiles_batch(
    smiles: Iterable[str], model: Union[None, nn.Module] = None, numpy: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    """
    Generate the embeddings for a batch of SMILES strings.

    Parameters
    ===========
    smiles : Iterable[str]
        An iterable of SMILES strings to embed

    model : Union[None, nn.Module], by default None
        An optional embedding model; by default uses
        the pretrained VICGAE model.

    numpy : bool, by default True
        Flag to return NumPy arrays, as opposed to
        a `torch.Tensor`
    """
    if not model:
        model = VICGAE.from_pretrained()
    vectors = torch.stack([model.embed_smiles(s) for s in smiles]).squeeze(1)
    if numpy:
        return vectors.numpy()
    return vectors


def embed_smiles(
    smiles: str, model: Union[None, nn.Module] = None, numpy: bool = True
) -> Union[torch.Tensor, np.ndarray]:
    """
    Generate the embeddings for a single SMILES string.

    Parameters
    ===========
    smiles : str
        The SMILES string to convert to an embedding

    model : Union[None, nn.Module], by default None
        An optional embedding model; by default uses
        the pretrained VICGAE model.

    numpy : bool, by default True
        Flag to return NumPy arrays, as opposed to
        a `torch.Tensor`
    """
    if not model:
        model = VICGAE.from_pretrained()
    vectors = model.embed_smiles(smiles).squeeze(0)
    if numpy:
        return vectors.numpy()
    return vectors
