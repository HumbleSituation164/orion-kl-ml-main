#!/usr/bin/env python

import numpy as np
import pandas as pd

from embedder import xclass, embedder
import selfies as sf
import joblib
from joblib import load

from astrochem_embedding import VICGAE

from typing import *
import matplotlib.pyplot as plt
import tqdm
import torch
import periodictable as pt

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import scipy.stats as ss
from rdkit.Chem import MolFromSmiles as smi2mol
from rdkit.Chem import MolToSmiles as mol2smi
from rdkit.Chem import rdchem
from rdkit.Chem import rdFMCS as MCS

import exmol
from exmol import stoned, Example
from exmol.exmol import _select_examples

# ========================================================

#vocab = pd.read_csv(r"/home/hnscolati/orion-kl-ml/data/okl_vocab1.csv")

# replacing CSV file with vocab from VICGAE
model = VICGAE.from_pretrained()
vocab = model.vocab
alphabet = set(vocab.alphabet)
filtered_alphabet = set()
for token in alphabet:
    # we want to strip the crap and just keep the element
    elemental_symbol = ''.join(list(filter(lambda x: x.isalpha(), token)))
    # check if we can get the element; if it's not a valid element throw it out
    element_obj = getattr(pt.elements, elemental_symbol, None)
    # only include elements below a certain mass; this is about calcium
    if element_obj is not None and getattr(element_obj, "mass") <= 42.:
        filtered_alphabet.add(token)


def compute_cos_similarity(reference_smiles: str, target_selfies: str) -> float:
    # convert smiles to embeddings, these are the A and B vectors
    a = embedder.embed_smiles(reference_smiles).numpy().squeeze()
    # embed SELFIES directly, so we don't have to do back-and-forth translation
    (encoding, _) = embedder.vocab.tokenize_selfies(target_selfies)
    tokens = torch.LongTensor(encoding)
    b = embedder.embed_molecule(tokens).numpy().squeeze()

    # find dot product
    ab_dot = np.dot(a, b.T)

    # calculate vector magnitudes 
    a_mag = np.linalg.norm(a)
    b_mag = np.linalg.norm(b)
    
    # calcualte the cosine similarity 
    cos_sim = ab_dot / (a_mag * b_mag)

    # return the cosine similarity 
    return cos_sim


def euclidean_distance(reference_smiles: str, target_selfies: str) -> float:
    # convert smiles to embeddings, these are the A and B vectors
    a = embedder.embed_smiles(reference_smiles).numpy().squeeze()
    # embed SELFIES directly, so we don't have to do back-and-forth translation
    (encoding, _) = embedder.vocab.tokenize_selfies(target_selfies)
    tokens = torch.LongTensor(encoding)
    b = embedder.embed_molecule(tokens).numpy().squeeze()
    inv_distance = 1. / np.sqrt((a - b)**2.).sum()
    if np.isnan(inv_distance):
        return 0.
    return inv_distance


def run_stoned(
    start_smiles: str,
    num_samples: int = 2000,
    max_mutations: int = 2,
    min_mutations: int = 1,
    alphabet: Union[List[str], Set[str]] = alphabet,
    return_selfies: bool = False,
    _pbar: Any = None,
) -> Union[Tuple[List[str], List[float]], Tuple[List[str], List[str], List[float]]]:
    """Run ths STONED SELFIES algorithm. Typically not used, call :func:`sample_space` instead.
    :param start_smiles: SMILES string to start from
    :param fp_type: Fingerprint type -> this was removed as no fingerprint will be used
    :param num_samples: Number of total molecules to generate
    :param max_mutations: Maximum number of mutations
    :param min_mutations: Minimum number of mutations
    :param alphabet: Alphabet to use for mutations, typically from :func:`get_basic_alphabet()`
    :param return_selfies: If SELFIES should be returned as well
    :return: SELFIES, SMILES, and SCORES generated or SMILES and SCORES generated
    """
    if alphabet is None:
        alphabet = get_basic_alphabet()
    if type(alphabet) == set:
        alphabet = list(alphabet)
    num_mutation_ls = list(range(min_mutations, max_mutations + 1))

    start_mol = smi2mol(start_smiles) # gets molecules from input smiles
    #print(start_mol)
    if start_mol == None:
        raise Exception("Invalid starting structure encountered")

    # want it so after sampling have num_samples
    randomized_smile_orderings = [
        stoned.randomize_smiles(smi2mol(start_smiles))
        for _ in range(num_samples // len(num_mutation_ls))
    ]

    # Convert all the molecules to SELFIES
    selfies_ls = [sf.encoder(x) for x in randomized_smile_orderings]

    all_smiles_collect: List[str] = []
    all_selfies_collect: List[str] = []
    for num_mutations in num_mutation_ls:
        # Mutate the SELFIES:
        if _pbar:
            _pbar.set_description(f"🥌STONED🥌 Mutations: {num_mutations}")
        selfies_mut = stoned.get_mutated_SELFIES(
            selfies_ls.copy(), num_mutations=num_mutations, alphabet=alphabet
        )
        # Convert back to SMILES:
        smiles_back = [sf.decoder(x) for x in selfies_mut]
        all_smiles_collect = all_smiles_collect + smiles_back
        all_selfies_collect = all_selfies_collect + selfies_mut
        if _pbar:
            _pbar.update(len(smiles_back))

    if _pbar:
        _pbar.set_description(f"🥌STONED🥌 Filtering")

    # filter out duplicates
    all_mols = [smi2mol(s) for s in all_smiles_collect]
    all_canon = [mol2smi(m, canonical=True) if m else None for m in all_mols]
    filter_mols, filter_selfies, filter_smiles, scores = [], [], [], []
    unique_smi = set()
    # loop over everything at the same time
    for i, (mol, selfie, smi) in enumerate(zip(all_mols, all_selfies_collect, all_canon)):
        if smi and smi not in unique_smi:
            # add to set so we don't count again
            unique_smi.add(smi)
            # add the strings to their lists
            filter_mols.append(mol)
            filter_smiles.append(smi)
            filter_selfies.append(selfie)
            # compute the similiarity score
            scores.append(
                euclidean_distance(start_smiles, selfie)
            )

    if _pbar:
        _pbar.set_description(f"🥌STONED🥌 Done")

    if return_selfies:
        return filter_selfies, filter_smiles, scores
    else:
        return filter_mols




def sample_space(
    origin_smiles: str,
    f: Union[
        Callable[[str, str], List[float]],
        Callable[[str], List[float]],
        Callable[[List[str], List[str]], List[float]],
        Callable[[List[str]], List[float]],
    ],
    batched: bool = True,
    preset: str = "medium",
    data: List[Union[str, rdchem.Mol]] = None,
    method_kwargs: Dict = None,
    num_samples: int = None,
    stoned_kwargs: Dict = None,
    quiet: bool = False,
    use_selfies: bool = False,
    sanitize_smiles: bool = True,
) -> List[Example]:
    """Sample chemical space around given SMILES
    This will evaluate the given function and run the :func:`run_stoned` function over chemical space around molecule. ``num_samples`` will be
    set to 3,000 by default if using STONED and 150 if using ``chemed``.
    :param origin_smiles: starting SMILES
    :param f: A function which takes in SMILES or SELFIES and returns predicted value. Assumed to work with lists of SMILES/SELFIES unless `batched = False`
    :param batched: If `f` is batched
    :param preset: Can be wide, medium, or narrow. Determines how far across chemical space is sampled. Try `"chemed"` preset to only sample commerically available compounds.
    :param data: If not None and preset is `"custom"` will use this data instead of generating new ones.
    :param method_kwargs: More control over STONED, CHEMED and CUSTOM can be set here. See :func:`run_stoned`, :func:`run_chemed` and  :func:`run_custom`
    :param num_samples: Number of desired samples. Can be set in `method_kwargs` (overrides) or here. `None` means default for preset
    :param stoned_kwargs: Backwards compatible alias for `methods_kwargs`
    :param quiet: If True, will not print progress bar
    :param use_selfies: If True, will use SELFIES instead of SMILES for `f`
    :param sanitize_smiles: If True, will sanitize all SMILES
    :return: List of generated :obj:`Example`
    """

    wrapped_f = f

    # if f only takes in 1 arg, wrap it in a function that takes in 2
    if f.__code__.co_argcount == 1:
        if use_selfies:

            def wrapped_f(sm, sf):
                return f(sf)

        else:

            def wrapped_f(sm, sf):
                return f(sm)

    batched_f: Any = wrapped_f
    if not batched:

        def batched_f(sm, se):
            return np.array([wrapped_f(smi, sei) for smi, sei in zip(sm, se)])

    if sanitize_smiles:
        origin_smiles = stoned.sanitize_smiles(origin_smiles)[1]
    if origin_smiles is None:
        raise ValueError("Given SMILES does not appear to be valid")
    smi_yhat = np.asarray(batched_f([origin_smiles], [sf.encoder(origin_smiles)]))
    try:
        iter(smi_yhat)
    except TypeError:
        raise ValueError("Your model function does not appear to be batched")
    smi_yhat = np.squeeze(smi_yhat[0])

    if stoned_kwargs is not None:
        method_kwargs = stoned_kwargs

    if method_kwargs is None:
        method_kwargs = {}
        if preset == "medium":
            method_kwargs["num_samples"] = 3000 if num_samples is None else num_samples
            method_kwargs["max_mutations"] = 2
            method_kwargs["alphabet"] = get_basic_alphabet()
        elif preset == "narrow":
            method_kwargs["num_samples"] = 3000 if num_samples is None else num_samples
            method_kwargs["max_mutations"] = 1
            method_kwargs["alphabet"] = get_basic_alphabet()
        elif preset == "wide":
            method_kwargs["num_samples"] = 3000 if num_samples is None else num_samples
            method_kwargs["max_mutations"] = 5
            method_kwargs["alphabet"] = sf.get_semantic_robust_alphabet()
        elif preset == "chemed":
            method_kwargs["num_samples"] = 150 if num_samples is None else num_samples
        elif preset == "custom" and data is not None:
            method_kwargs["num_samples"] = len(data)
        else:
            raise ValueError(f'Unknown preset "{preset}"')
    try:
        num_samples = method_kwargs["num_samples"]
    except KeyError as e:
        if num_samples is None:
            num_samples = 150
        method_kwargs["num_samples"] = num_samples

    pbar = tqdm.tqdm(total=num_samples, disable=quiet)

    # STONED
    if preset.startswith("chem"):
        smiles, scores = run_chemed(origin_smiles, _pbar=pbar, **method_kwargs)
        selfies = [sf.encoder(s) for s in smiles]
    elif preset == "custom":
        smiles, scores = run_custom(
            origin_smiles, data=cast(Any, data), _pbar=pbar, **method_kwargs
        )
        selfies = [sf.encoder(s) for s in smiles]
    else:
        result = run_stoned(
            origin_smiles, _pbar=pbar, return_selfies=True, **method_kwargs
        )
        selfies, smiles, scores = cast(Tuple[List[str], List[str], List[float]], result)

    pbar.set_description("😀Calling your model function😀")
    if sanitize_smiles:
        smiles = [stoned.sanitize_smiles(s)[1] for s in smiles]
        print(type(smiles))
    fxn_values = batched_f(smiles, selfies)

    # pack them into data structure with filtering out identical
    # and nan
    exps = [
        Example(
            origin_smiles,
            sf.encoder(origin_smiles),
            1.0,
            cast(Any, smi_yhat),
            index=0,
            is_origin=True,
        )
    ]
    for i, (smi, selfie, score, y) in enumerate(zip(smiles, selfies, scores, fxn_values)):
        exps.append(
            Example(smi, selfie, score, y, index=i)
        )

    # pbar.reset(len(exps))
    # pbar.set_description("🔭Projecting...🔭")

    # # compute distance matrix
    # full_dmat = _fp_dist_matrix(
    #     [e.smiles for e in exps],
    #     method_kwargs["fp_type"] if ("fp_type" in method_kwargs) else "ECFP4",
    #     _pbar=pbar,
    # )

    # pbar.set_description("🥰Finishing up🥰")

    # # compute PCA
    # pca = PCA(n_components=2)
    # proj_dmat = pca.fit_transform(full_dmat)
    # for e in exps:  # type: ignore
    #     e.position = proj_dmat[e.index, :]  # type: ignore

    # # do clustering everywhere (maybe do counter/same separately?)
    # # clustering = AgglomerativeClustering(
    # #    n_clusters=max_k, affinity='precomputed', linkage='complete').fit(full_dmat)
    # # Just do it on projected so it looks prettier.
    # clustering = DBSCAN(eps=0.15, min_samples=5).fit(proj_dmat)

    # for i, e in enumerate(exps):  # type: ignore
    #     e.cluster = clustering.labels_[i]  # type: ignore

    pbar.set_description("🤘Done🤘")
    pbar.close()
    return exps



# def _select_examples(cond, examples, nmols):
#     result = []

#     # similarity filtered by if cluster/counter
#     def cluster_score(e, i):
#         return (e.cluster == i) * cond(e) * e.similarity

#     clusters = set([e.cluster for e in examples])
#     for i in clusters:
#         close_counter = max(examples, key=lambda e, i=i: cluster_score(e, i))
#         # check if actually is (since call could have been zero)
#         if cluster_score(close_counter, i):
#             result.append(close_counter)

#     # trim, in case we had too many cluster
#     result = sorted(result, key=lambda v: v.similarity * cond(v), reverse=True)[:nmols]

#     # fill in remaining
#     ncount = sum([cond(e) for e in result])
#     fill = max(0, nmols - ncount)
#     result.extend(
#         sorted(examples, key=lambda v: v.similarity * cond(v), reverse=True)[:fill]
#     )

#     return list(filter(cond, result))


def cf_explain(examples: List[Example], nmols: int = 3) -> List[Example]:
    """From given :obj:`Examples<Example>`, find closest counterfactuals (see :doc:`index`)
    :param examples: Output from :func:`sample_space`
    :param nmols: Desired number of molecules
    """

    def is_counter(e):
        return e.yhat != examples[0].yhat

    result = _select_examples(is_counter, examples[1:], nmols)
    for i, r in enumerate(result):
        r.label = f"Counterfactual {i+1}"

    return examples[:1] + result






# note, this function processes one smiles at a time: whatever is given as the base
def get_colDen(smiles):
    # imports embedder and converts given smiles to selfies, then tokenizes into vectors
    embeddings = embedder.embed_smiles(smiles).numpy()

    # hard code parameters
    # t_rot, v_lsr, dv, isotopologue, environment code, excited state
    features = np.array([[100, 7.2, 10, 0, 1, 0]])

    # combines np.arrays to correct dim for ML model
    combined = np.hstack([features, embeddings])

    # predicts column density, returns 1D nd.array
    Ncol = gradient_boosting.predict(combined)

    return Ncol    


def fake_objective_function(smi: str) -> float:
    """This is to test the rest of the STONED pipeline without having a real model"""
    return np.random.rand()


# =============================================================

#load in desired model
gradient_boosting = joblib.load('gradientboosting.pkl')

# vocab = pd.read_csv (r'/home/hnscolati/orion-kl-ml/data/okl_vocab1.csv')
# alphabet = set(vocab.values.ravel())


# print(compute_tanimoto_similarity('COC', 'C[17O+]OC'))
# print(compute_cos_similarity('COC', 'C[17O+]OC'))


base = 'COC'
x = run_stoned(base)
samples = sample_space(
    base,
    fake_objective_function,
    batched=False,
    use_selfies=True,
    method_kwargs={"alphabet": filtered_alphabet},
)
cfs = cf_explain(samples, 5)
print(cfs)

exmol.plot_cf(cfs)
plt.show()
