"""Run experiments not in parallel, only one trial and only a subset of them in
order to allow for debugging."""
from typing import List

from rsp.hypothesis_one_data_analysis import hypothesis_one_data_analysis
from rsp.hypothesis_one_experiments import get_first_agenda_pipeline_params
from rsp.utils.experiments import create_experiment_agenda
from rsp.utils.experiments import run_experiment_agenda


def _run_experiment_ids_from_agenda(experiment_ids: List[int]):
    parameter_ranges, speed_data = get_first_agenda_pipeline_params()
    # Create an experiment agenda out of the parameter ranges
    experiment_agenda = create_experiment_agenda(experiment_name="exp_hypothesis_one",
                                                 speed_data=speed_data,
                                                 parameter_ranges=parameter_ranges,
                                                 trials_per_experiment=1)

    # Run experiments
    experiment_folder_name = run_experiment_agenda(
        experiment_agenda=experiment_agenda,
        experiment_ids=experiment_ids,
        run_experiments_parallel=False,
        show_results_without_details=True,
        verbose=False)
    hypothesis_one_data_analysis(
        data_folder=experiment_folder_name,
        analysis_2d=True,
        analysis_3d=False,
        malfunction_analysis=False,
        qualitative_analysis_experiment_ids=experiment_ids)


if __name__ == '__main__':
    experiment_ids = [12]
    _run_experiment_ids_from_agenda(experiment_ids)
