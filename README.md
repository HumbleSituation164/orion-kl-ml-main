orion-kl-ml
==============================

ML analysis of Orion KL

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

## Developer workflow

Because we're working in a team, we should adopt some reasonably good practices with sharing
a common codebase. Primarily this means no peeing in the pool, and putting code into functions
that others can use with reasonable documentation.

Since we're using git, we can adopt some good practices quite easily: work off your own branch,
and when someone has made code contributions, submit a pull request; here's the workflow:

1. Fork the repository to your account, and then clone it to your computer.
1. After setting everything up, run `git branch -b name-of-branch`; for ease I would just use your name.
    - This means that everything you do is spinning off the `main` branch, and you won't accidentally add things to it.
2. Install `git` pre-hooks using `pre-commit install`
2. Write some code in `orion-kl`, make modules, tests, whatever. Add these changes to the `git` tracking with `git add all-the-files-you-want-to-add`.
3. Commit your changes, and commit often. Run `git commit` and it'll bring up a text editor that you can write:
    - A short "title" message describing what's been done at a glance.
    - A longer optional message that goes into more detail.
4. If the hooks are installed correctly, it will run some checks; if your commit stops abruptly, it means something was changed by hooks, probably `black` formatting. Just add the files changed and re-commit.
4. Run `git push` to send your changes up to your github repository.
5. Go to your repository on the web, and a "Pull Request" button should appear.
