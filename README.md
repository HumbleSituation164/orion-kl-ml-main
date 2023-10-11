**NOTE: this repository is currently under construction**

Explaining chemical inventories with machine learning
==============================

![Static Badge](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)
![Static Badge](https://img.shields.io/badge/license-MIT-green)
![Static Badge](https://img.shields.io/badge/pypi-v0.2.0-yellow)
![Static Badge](https://img.shields.io/badge/code%20style-black-black)



This repository includes the codebase for the machine leanring pipelines developed and applied to predicting chemical abundances of unobserved interstellar species. This work focuses on the Orion Kleinmann-Low (Orion KL) nebula and its respective environments. 

The corresponding [dataset](https://doi.org/10.5281/zenodo.7675609) can be downloaded via zenodo. If used, please cite accordingly. 

If you used the results generated from this work as part of your own research, please cite the [zenodo repository](https://doi.org/10.5281/zenodo.7675609) and the paper once published. In the meantime, please cite this repository. 

## Requirements
This package requires Python 3.8+, as the embedding package uses some decorators only available after 3.7.

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

This uses the newly developed seq2seq model, `VICGAE`. Here is a quick installation and use guide to get you started, but see the [repo](https://github.com/laserkelvin/astrochem_embedding)
for more details if you're interested in development or training your own model.

### VICGAE molecular embedder installation and usage

`astrochem_embedding` can be installed quickly through PyPI:
```python
$ pip install astrochem_embedding
```

You can check the environment using a pre-trained model:
```python
>>> from astrochem_embedding import VICGAE
>>> import torch
>>> model = VICGAE.from_pretrained()
>>> model.embed_smiles("c1ccccc1")
```

===========================

## Instructions


### Predicting column denisties


### Generating counterfactuals

This step uses a modified version of the exmol package developed by the White group. Please see their [publication](https://doi.org/10.1039/D1SC05259D ) and [repository](https://github.com/ur-whitelab/exmol) for more details. 


## Project Structure

--------
## License
Distributed under the terms of the [MIT license](https://opensource.org/license/mit/)

## Issues

## Credits
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
This version of the cookiecutter template is modified by Kelvin Lee.
