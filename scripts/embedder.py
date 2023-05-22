#!/usr/bin/env python


import pandas as pd
from astrochem_embedding import VICGAE
import torch

from common import smiles_check

#==========================================================================

pd.set_option('display.max_rows', None, 'display.max_column', None)

xclass = pd.read_csv (r'/home/hnscolati/orion-kl-ml/data/raw/orionkl_parameters_xclass.csv')

#xclass.set_index('Environment', inplace=True)


embedder = VICGAE.from_pretrained()
xclass['SMILES vector'] = ''
#print(smiles_check(xclass, embedder)[0])

xclass = smiles_check(xclass, embedder)[0]
print(xclass)

#smiles_check(xclass, embedder)[0]
