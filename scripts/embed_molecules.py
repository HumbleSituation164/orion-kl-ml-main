import numpy as np
import pandas as pd
from joblib import dump

import orion_kl_ml as okl


mol_df = pd.read_csv(okl.paths.get("raw").joinpath("orionkl_parameters_xclass.csv"))

# get only the unique SMILES strings
smiles = sorted(list(set(mol_df["SMILES"].values)))

embeddings = {}

for smi in smiles:
    z = okl.embed_smiles(smi)
    embeddings[smi] = z

target = okl.paths.get("processed")
dump(embeddings, target.joinpath("embeddings.pkl"))
