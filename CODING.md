# RSP Coding Guidelines and Design Decisions


### Python and conda
Since the project should be accessible to developers and (data) scientists, we use `python` as compromise
- type hints: these improve readability and give better IDE support (code completion and static checking)
- named tuples (for lack of better alternative): this gives data structure fields a name, which makes the code more accessible; high-performance in our code has not been a key goal so far.

Caveats:
- We sometimes generate the fields of named tuples; this may hamper code completion again.
- We refrained from using `mypy` for static type checking, as it is rather picky and ideosyncratic at times, which may impede non-developers too much.

### Linting
We use `pre-commit` git hook as a wrapper for static code analysis tools (`flake8`, `black` etc.) and formatting, see `.pre-commit-config.yaml` for the configuration.
This allows to check conformity before commiting and enforcing it in continuous integration (see below).



### pkl and csv
We use a three-level approach:
- pkl with `ExperimentResult`
- `load_and_expand_experiment_results_from_data_folder` loads this pkl and expands into `ExperimentResultsAnalysis`.
- `float` fields

We use pickle in order not to lose information as with JSON export/import.
We use `Pandas` csv export and import for floating data only in order to save space and in order to speed up import in Jupyter notebooks.


For renamings of data structures, there is a mapping:
`src/python/rsp/utils/rename_unpickler.py`
This approach does not work for renaming of fields of NamedTuples or for changes of structure.

Recommendations
- If you load many experiments, use `nonify_all_structured_fields` in `load_and_expand_experiment_results_from_data_folder` to set all non-number fields to `None`; without, all data is hold in memory!
- Use hierarchy `infra -> schedule -> re-scheduling` to only re-reun the parts you make breaking changes on!
- When working on analysis of data only, pkl contains all raw data and therefore allows to develop analysis of this data as a subsequent step

Notebooks:
- `compute_time_analysis.Rmd`: requires only csv, can re-import for pkl
- `detailed_experiment_analysis.Rmd`: requires pkl

### plotly
We currently use `plotly` for visualization. Use the generic helpers. Do we need more for diagrams in publication?

### package structure
We use one top-level package per pipeline step and one for the generic solver model:

    hypothesis_one_experiments_potassco.py    <- main entry point for running experiments
    hypothesis_one_malfunction_experiments.py <- do we still need this?
    hypothesis_one_pipeline_all_in_one.py     <- should this go under utils?
    scheduling
    step_01_planning
    step_02_setup
    step_03_run
    step_04_analysis
    transmission_chains                        <- should this go under step_03_run?
    utils                                      <- resource_occupations here?

We put data structures and their accessors and modifiers and transformers and validators in a separate file. The file is placed in the first step where it occurs. (See whishlist below.)




### Code in Jupyter notebooks

Recommendations:
- Use only method calls in Jupyter notebooks, no loops and coding.
- Run and test these methods in unit tests; we try to enforce at least running this by `vulture` in `pre-commit`.
- Jupyter notebooks are currently not run in continuous integration, could be re-enabled, see `tests/04_notebooks/test_notebook_runs_through.py`, use only a subset of data for this.


### `NumPy` Docstrings
In IntelliJ, set it under `Settings -> Tools -> Python Integrated Tools -> Docstrings`.
See https://numpydoc.readthedocs.io/en/latest/format.html for details about this format.


### Continuous Integration
We use SBB infrastructure for continuous integration, see https://confluence.sbb.ch/x/egWEWQ.
We do not have free infrastructure on github or gitlab.

### Testing

There are two types of assertions we want to test:

- functional: is the output as expected, complete or partial (e.g. does the solution have the expected costs without checking the full solution? Is the expected number of files generated?)
- non-functional: Does the code run through without error? Does it not run longer than expected?

We use four testing levels:

    01_unit_tests        <-- functional verification at the function or class level (sub-step level)
    02_regression_tests  <-- functional or non-functional verification at the module level (step or sub-step)
    03_pipeline_tests    <-- functional or non-functional verification at the pipeline level
    04_notebooks         <-- functional or non-functional verification at the notebook level

Test files should have the same name as the corresponding unit that is being tested.

Details:  https://confluence.sbb.ch/x/MpFlVg


### Overleaf
Overleaf sources should be manually synced into this repository. See description [README.md](doc/overleaf/README.md)


## Wishlist


### Cleanup apidoc (https://issues.sbb.ch/browse/SIM-723)
Apidoc is not consistently maintained (missing or empty parameter descriptions, outdated descriptions).
Furthermore, the generated apidoc should be inspected, not only in the source code, without too many redundancies to overleaf.

### Extract verifications (https://issues.sbb.ch/browse/SIM-324)
We often use Betrand-Meyer style  preconditions, postconditions and validators for data structure invariants:
- Applying "Design by Contract", Bertrand Meyer, IEEE Computer 1992. <http://se.ethz.ch/~meyer/publications/computer/contract.pdf>
- <https://docs.oracle.com/javase/7/docs/technotes/guides/language/assert.html>

Recommendations:
- simplify the validations (there are often too many cases and even obsolete cases covered)
- extract as much into unit testing as possible
- profile code to see how much time goes into these validations
- is it good practice to use plain `asserts` for this purpose or should we use specific exceptions? See <https://confluence.sbb.ch/pages/viewpage.action?pageId=1177719206#BestPracticesfor%22JavaCodeConventionsSBBv4%22-Don'tuseAsserts> in the context of JAVA
- discuss whether we should completely get rid of this coding style? While developing algoriths, it often proved invaluable!

### Cleanup unit tests (https://issues.sbb.ch/browse/SIM-323)
- During bugfixing, too large data was checked in into unit tests. This requires much scrolling and is not refactoring safe!
- There is not enough testing at the unit test level. This violates the testing pyramid!

### Consistent naming (https://issues.sbb.ch/browse/SIM-348)
- Many different data structure containing schedule, give more descriptive name
- Agent vs. train?
- Source vs. start and target vs. sink?


### Better abstraction and naming for `ScheduleProblemDescription` (https://issues.sbb.ch/browse/SIM-746)
- Make `ScheduleProblemDescription` closer to the description in the text
- Better name for `RouteDAGConstraints`?
- Better name for `SchedulingExperimentResult` (it's not an experiment in the agenda sense!)

### Encapsulate solver model (https://issues.sbb.ch/browse/SIM-121)
We currently use sets of strings for the ASP predicates and parse these strings. When coming from the solver, the answer sets have structure we throw away.

### Data structures (https://issues.sbb.ch/browse/SIM-748)
A cleaner approach would be to use data structure only within a top-level package:
Data structures are either
* internal to top-level package
* passed from step i to step i+1
* dedicated exposed data structure of other top-level package such as scheduling
