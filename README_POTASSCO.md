Instructions for Potassco
=========================

Workflow SBB <-> Potassco.

Installation
------------

```
# create conda environment rsp
conda env create -f rsp_environment.yml

# activate the conda env (if you want to use a different env name, run conda env create -f rsp_environment.yml --name other-env-name)
conda activate rsp
```

Export facts and call for an experiment
---------------------------------------

We store experiments in `git@github.com:SchweizerischeBundesbahnen/rsp-data.git`. We assume that the two repositories are cloned to the same folder:

```

$WORKSPACE_FOLDER/rsp        # <---- your local clone of git@github.com:SchweizerischeBundesbahnen/rsp.git
$WORKSPACE_FOLDER/rsp-data   # <---- your local clone of git@github.com:SchweizerischeBundesbahnen/rsp-data.git

```

```shell script
conda activate rsp
cd $WORKSPACE_FOLDER/rsp
export PYTHONPATH=$PWD:$PYTHONPATH
python rsp/asp_plausibility/potassco_export.py --experiment_base_directory=../rsp-data/many_agent_example --experiment_id=0 --problem=full_after_malfunction
```

Modifiy encoding
----------------
Modify `res/asp/encodings/*.lp`


Check correctness
-----------------
To run the modified program with the data und verify correctness:
```shell script
conda activate rsp
cd $WORKSPACE_FOLDER/rsp
export PYTHONPATH=$PWD:$PYTHONPATH
python rsp/asp_plausibility/potassco_solution_checker.py  --experiment_data_folder_name=../rsp-data/many_agent_example/data --experiment_id=0 --problem=full_after_malfunction
```

The correcntess checker is implemented here: `rsp/experiment_solvers/asp/asp_solution_description.py#verify_correctness_helper()`

### Pre-commit hook
In order to run pre-commit hooks when you run `git commit` on the command line
```
conda activate rsp
conda install -c conda-forge pre-commit
pre-commit install

# test the pre-commit
pre-commit run --all

# now run git commit with pre-commit
git commit -am 'my commit message' && git push -u
```
The pre-commit is only run on the files changed.

Details:
* [ pre-commit.  A framework for managing and maintaining multi-language pre-commit hooks.](https://pre-commit.com/)