**NOTE: this repository is currently under construction**

Explaining chemical inventories with machine learning
==============================

![Static Badge](https://img.shields.io/badge/python-3.8%20%7C%203.9-blue)
![Static Badge](https://img.shields.io/badge/pypi-v0.2.0-yellow)
![Static Badge](https://img.shields.io/badge/license-MIT-green)
![Static Badge](https://img.shields.io/badge/code%20style-black-black)



This repository includes notebooks and the codebase for the machine learning pipelines developed for the application of predicting chemical abundances of unobserved interstellar species for the Orion Kleinmann-Low (Orion KL) nebula and its respective environments. 

The primary focus of this work is generating base predictions of column densities for targets of interest for molecular surveys. Inspiration was derived from [Lee et al. (2021)](https://iopscience.iop.org/article/10.3847/2041-8213/ac194b/meta) to apply similar supervised and unsupervised methodologies to a source with more physical and chemical complexity. Here we use a similar workflow as Lee+ 2021, with modifications made to the embedding pipeline. Installation and workflow instructions are provided below. Full working examples are provided as notebooks accessible through this repo. 

The corresponding [dataset](https://doi.org/10.5281/zenodo.7675609) can be downloaded via Zenodo. If used, please cite accordingly. If you use the results generated from this work as part of your own research, please cite the [Zenodo repository](https://doi.org/10.5281/zenodo.7675609) and the accepted paper once published (stay tuned!). In the meantime, please cite this repository. 

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

Users can generate their own predictions, however it is manual at this time. As part of the repository, we've included a pretrained embedding model, as well as the two regressors stored as pickles dumped using joblib. With some modification, other regressors available via scikit-learn packages can be trained using our pipeline.  

## Project workflow

Our goal was to make this project modular so that the pieces (embedder, regressor, dataset) could be swapped out to fit the users' needs. 

### Molecule embedding

1. Collect all SMILES strings for your molecules and put them into a single `.csv` file. Alternatively, the `VICGAE` embedder also accepts SELFIES string, so you could compile your `.csv` file as SELFIES instead, if you prefer.
2. Transform your SMILES strings into vectors using the embedding pipeline. The script `scripts/embed_molecules.py` contains the functions you will need to do this, or you can check out the notebook for a working example.
   Note: if you would like to use the pretrained `VICGAE` embedder, you can fork the repo at [laserkelvin/astrochem_embedding](https://github.com/laserkelvin/astrochem_embedding).

### Training the regressors

With the embedding pipeline set up and the molecular embedding vectors acquired, we can now train a regressor to predict the column density of your molecule (or molecules) of choice. We advise you set up a `.csv` file or other machine readable format that holds all of the molecules, their embeddings, column densities, and other physical features as needed (e.g. velocity components, line widths). The codebase is available in the script `scripts/train_models.py`, or you can use one of the pretrained regressors stored as a pickle under `models/`. You can also train (with modification) other regressors available through scikit-learn packages. Our notebook provides a working example if you would like to go this route.

### Predicting column denisties

Once you have a trained regressor of choice, you can compile your molecules of interest into a `.csv` file with their SMILES and/or SELFIES strings. The script `scripts/make_predictions.py` will generate the simulated parameters for your targets to generate your predictions for a chosen region in Orion KL.

### Generating counterfactuals

This step uses a modified version of the `exmol` package developed by the White group. Please see their [publication](https://doi.org/10.1039/D1SC05259D ) and [repository](https://github.com/ur-whitelab/exmol) for more details. This step isn't necessary, but does provide a chance to make some interesting constrastive examples to explore the interpretbility of detectibility for molecules. Essentially, this modified script will take a molecule of choice, make minor mutations to the base structure via addition, subtraction, or swapping, and generate that new counterfactual structure. This first portion does not neccessarily need our pipeline, however, `exmol` can output the corresponding SELFIES string to feed into a trained regressor and predict its column density. This method, although in it's infancy, can provide a means of a testable hypothesis in looking at correlations between functional groups and molecular structural components with thier abundance. 


## Project Structure

├── CITATION.cff

├── LICENSE

├── data

│   ├── processed

│   └── raw

├── models

│   ├── gbr.pkl

│   ├── rfr.pkl

│   ├── translator.yml

│   └── translator_no_iso.yml

├── notebooks

│   ├── dev

│   ├── exploratory

│   └── reports

├── poetry.lock

├── pyproject.toml

├── README.md

├── scripts

│   ├── embed_molecules.py

│   └── make_counterfactuals.py

└── src

   └── orion_kl_ml
   
      ├── __init__.py
      
      ├── models
      
      │   ├── __init__.py
      
      │   ├── base.py
      
      │   ├── embed.py
      
      │   ├── layers.py
      
      │   └── tests
      
      │       ├── __init__.py
      
      │       └── test_embed.py
      
      ├── pipeline
      
      │   ├── data.py
      
      │   ├── __init__.py
      
      │   ├── tests
      
      │   │   ├── __init__.py
      
      │   ├── make_dataset.py
      
      │   └── transforms.py
      
      ├── visualization
      
      │   ├── __init__.py
      
      │   └── visualize.py
      
      ├── __main__.py
      
      └── utils.py
   

--------
## License
Distributed under the terms of the [MIT license](https://opensource.org/license/mit/)

## Issues

## Credits
<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
This version of the cookiecutter template is modified by Kelvin Lee.
