# Real Time Large Network Re-Scheduling

## Quick Start Installation
You need to have installed `conda` and `git` (temporarily for FLATland). Then,

In order to run the experiments,
```shell

# create conda environment rsp
conda env create -f rsp_environment.yml

# activate the conda env (if you want to use a different env name, run conda env create -f rsp_environment.yml --name other-env-name)
conda activate rsp

#
export PYTHONPATH=$PWD/src/python:$PWD/src/asp:$PYTHONPATH

# run pipeline (see section "Getting Started with Experiments" below)
python src/python/rsp/hypothesis_one_experiments_potassco.py

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

## Getting Started with Experiments
You should only need two files:
* `src/python/rsp/hypothesis_one_experiments_potassco.py`: defines parameters of your experiments
* `src/python/rsp/utils/global_data_configuration.py`: defines data location
Here, you can define parameters on three levels:

 level         | parameter range data structure  | defined in                                 | deterministic (yes/no)
 --------------|---------------------------------|--------------------------------------------|--------------
infrastructure | `InfrastructureParametersRange` | `generate_potassco_infras_and_schedules`   | yes
schedule       | `ScheduleParametersRange`       | `generate_potassco_infras_and_schedules`   | no
re-schedule    | `ScheduleParametersRange`       | `run_potassco_agenda`                      | no

The Cartesian product of parameter settings at all three levels defines an experiment agenda of experiments. The infrastructure level is deterministic, the schedule and re-schedule level are not deterministic;
examples:
 * if you choose three sizes and two seeds for infrastructure generation, this will produce 6 infrastructure
 * if you choose two seeds for schedule generation, this will produce 2 schedules for *each* infrastructure
 * if you choose two seeds for re-scheduling, this will run two re-scheduling experiments for every pair of infrastructure and schedule

In addition, it is possible to run multiple agendas with different solver settings (these are not part of parameter ranges, since these settings are categorical and not quantitative).

The data layout will look as follows:

    .
    ├── h1_2020_10_08T16_32_00
    │   ├── README.md
    │   ├── h1_2020_10_08T16_32_00_baseline_2020_10_14T19_06_38
    │   │   ├── data
    │   │   │   ├── err.txt
    │   │   │   ├── experiment_2064_2020_10_14T19_07_39.pkl
    │   │   │   ├── experiment_2067_2020_10_14T19_07_41.pkl

    │   │   │   ├── experiment_2961_2020_10_15T05_49_18.pkl
    │   │   │   ├── experiment_3056_2020_10_15T06_18_14.pkl
    │   │   │   └── log.txt
    │   │   ├── experiment_agenda.pkl
    │   │   └── sha.txt
        └── infra
    │       └── 000
    │           ├── infrastructure.pkl
    │           ├── infrastructure_parameters.pkl
    │           └── schedule
    │               ├── 000
    │               │   ├── schedule.pkl
    │               │   └── schedule_parameters.pkl
    │               ├── 001
    │               │   ├── schedule.pkl
    │               │   └── schedule_parameters.pkl
    │


Here's the main part of `src/python/rsp/hypothesis_one_experiments_potassco.py`:

    generate_potassco_infras_and_schedules(base_directory=INFRAS_AND_SCHEDULES_FOLDER, parallel_compute=parallel_compute)
    run_potassco_agenda(
        base_directory=INFRAS_AND_SCHEDULES_FOLDER,
        # incremental re-start after interruption
        experiment_output_base_directory=BASELINE_DATA_FOLDER,
        experiment_filter=experiment_filter_first_ten_of_each_schedule,
        parallel_compute=parallel_compute,
        csv_only=False,
    )

It consists of infrastructure and schedule generation (`infra` subfolder) and one or more agenda runs  (`h1_2020_08_24T21_04_42_baseline_2020_10_12T13_59_59` subfolder for the run with baseline solver settings).

See the use cases below for details how to use these options of `generate_potassco_infras_and_schedules` and `run_potassco_agenda`.


### Use case 1a: run all three levels
Configure `INFRAS_AND_SCHEDULES_FOLDER` in `src/python/rsp/utils/global_data_configuration.py` to point to the base directory for your data.

    INFRAS_AND_SCHEDULES_FOLDER = "../rsp-data/h1_2020_10_08T16_32_00"

Infrastructures will be generated into a subfolder `infra` under this base folder.
In addition, if you comment out the argument

    # experiment_output_base_directory=...

the experiment agenda will also get a new timestamped subfolder here; if you uncomment the argument, you will need to define

    BASELINE_DATA_FOLDER = "../rsp-data/h1_2020_10_08T16_32_00/h1_2020_10_08T16_32_00_baseline_2020_10_21T18_16_25"

appropriately.

### Use case 1b: only use baseline solver settings
Comment out calls to `list_from_base_directory_and_run_experiment_agenda` you don't need.

### Use case 2a: you've aborted scheduling and want to run experiments on the schedules you already have
Comment out `generate_potassco_infras_and_schedules(...)` in `main`. The agenda will only contain experiments for the existing schedules.

### Use case 2b: you've aborted experiments and want to run a certain subset of experiments into the same data folder
Configure `BASELINE_DATA_FOLDER` in `src/python/rsp/utils/global_data_configuration.py` to point to the location you want to have your experiments in;
this will be a subfolder of your base directory for data. In order to apply a filter on the agenda, you will need to give a different filter:

    def experiment_filter_first_ten_of_each_schedule(experiment: ExperimentParameters):
        return experiment.re_schedule_parameters.malfunction_agent_id < 10 and experiment.experiment_id >= 2000

    if __name__ == "__main__":
        ...
        run_potassco_agenda(
            ...
            experiment_filter=experiment_filter_first_ten_of_each_schedule,
            ...
        )
    experiment_filter=experiment_filter_first_ten_of_each_schedule

### Use case 2c: you want to generate more schedules after you've already run experiments
You will need to tweak:
* Define an appropriate filter (see `experiment_filter_first_ten_of_each_schedule`) for the missing experiments (they will have larger experiment ids than so far)
* Run scheduling with the same `INFRAS_AND_SCHEDULES_FOLDER` as before; this will add the missing schedules incrementally;
* Use a new `BASELINE_DATA_FOLDER` for running the experiments. Be sure you use the same parameters as before.
* Copy the older experiments to the new location.

## Coding Guidelines
See [CODING.md](CODING.md).

## Disclaimer
### Authors:
- Adrian Egli
- Christian Eichenberger
- Erik Nygren
