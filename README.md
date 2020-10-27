# Real Time Large Network Re-Scheduling

## Quick Start
You need to have installed `conda` and `git` (temporarily for FLATland). Then,

In order to run the experiments,
```shell

# create conda environment rsp
conda env create -f rsp_environment.yml

# activate the conda env (if you want to use a different env name, run conda env create -f rsp_environment.yml --name other-env-name)
conda activate rsp

#
export PYTHONPATH=$PWD/src/python:$PWD/src/asp:$PYTHONPATH

# run pipeline
python src/python/rsp/hypothesis_one_experiments.py

# ..... do some development....

# (optionally) update the conda env if rsp_environment.yml was modified
conda env update -f rsp_environment.yml

# run tests
python -m pytest
```
### Setup Jupyter Notebooks
Taken from [this](https://towardsdatascience.com/version-control-with-jupyter-notebooks-f096f4d7035a) post,
this is a short introduction on how to use Jupyter Notebooks with git.

Start by installing the jupytext extensions
```
pip install jupytext --upgrade
```
and make the accessible by your notebooks in the conda env by installing ([Guide](https://stackoverflow.com/questions/37433363/link-conda-environment-with-jupyter-notebook))
```
conda install nb_conda
```

Generate a Jupyter config, if you don’t have one yet, with ```jupyter notebook --generate-config```
edit ```.jupyter/jupyter_notebook_config.py``` and append the following:
```
c.NotebookApp.contents_manager_class="jupytext.TextFileContentsManager"
c.ContentsManager.default_jupytext_formats = ".ipynb,.Rmd"
```
and restart Jupyter, i.e. run
```
jupyter notebook
```
Note: ```.jupyter``` is mostly present in your home directory.

Open an existing notebook or create a new one.
Disable Jupyter’s autosave to do round-trip editing, just add the following in the top cell and execute.
```
%autosave 0
```
You can edit the ```.Rmd``` file in Jupyter as well as text editors, and this can be used to check version control changes.

### Cloning the repo and create notebook
Open the ```.Rmd``` file in jupyter from its file browser.
You can use the ```.Rmd``` file directly but it will not persist output between sessions, so we are gonna create a jupyter notebook.

- Click File->Save (Cmd/Ctrl+S).
- Close the ```.Rmd``` file (File->Close and Halt)

Now open the ```.ipynb``` in Jupyter.
Start editing and saving. Your ```.Rmd``` file will keep updating itself.

### Pre-commit hook
In order to run pre-commit hooks when you run `git commit` on the command line
```
conda activate rsp
conda install -c conda-forge pre-commit
pre-commit install

# test the pre-commit
pre-commit run --all
```
The pre-commit is only run on the files changed.

Details:
* [ pre-commit.  A framework for managing and maintaining multi-language pre-commit hooks.](https://pre-commit.com/)

### Automatic mpeg conversion of FLATland
In order to have automatic mpeg conversion, we use the python-wrapper [ffmpeg-python](https://github.com/kkroening/ffmpeg-python/blob/master/examples/README.md).
For this to work, `ffmpeg` must be installed and on the `PATH`.


## Coding Guidelines
See [CODING.md](CODING.md).

## Disclaimer
### Authors:
- Adrian Egli
- Christian Eichenberger
- Erik Nygren
