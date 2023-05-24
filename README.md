orion-kl-ml
==============================

This repository includes the codebase for the machine leanring pipelines developed and applied to predicting chemical abundances of unobserved species in the Orion KL nebula. 

If you used the results generated from this work as part of your own research, please cite the zenodo repository <a href="https://doi.org/10.5281/zenodo.7675609"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7675609.svg" alt="DOI"></a> and the paper once published. In the meantime, please cite this repository. 

## Installation

1. Use `conda` to install from the YAML specification with `conda env create -f conda.yml`
2. Activate the environment by typing `conda activate orion-kl`
3. Install the Python requirements using `poetry install`

You can test that the environment is working by running `ipython` and trying:

```python
>>>from orion_kl_ml import embed_smiles, embed_smiles_batch
# by default returns a NumPy array
>>> embed_smiles("c1ccccc1")
# if you want to work with torch.Tensors
>>> embed_smiles("c1ccccc1", numpy=False)
# operate on a list of SMILES
>>> smiles = ["c1ccccc1", "CC#N", "C#CC#CC#CC"]
>>> embed_smiles_batch(smiles)
```

This uses the newly developed seq2seq model, `VICGAE`, with details forthcoming. See the [repo](https://github.com/laserkelvin/astrochem_embedding)
for more details.

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
This version of the cookiecutter template is modified by Kelvin Lee.

## Instructions
