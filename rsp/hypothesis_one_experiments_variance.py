"""Run experiments not in parallel, only one trial and only a subset of them in
order to allow for debugging."""
import numpy as np

from rsp.hypothesis_one_data_analysis import hypothesis_one_data_analysis
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.experiments import run_experiment_agenda


def _run_experiment_ids_from_agenda():
    nb_experiments = 100
    experiment_agenda = ExperimentAgenda(
        experiment_name="",
        experiments=[
            ExperimentParameters(experiment_id=0, grid_id=0, number_of_agents=5,
                                 width=30, height=30,
                                 flatland_seed_value=11, asp_seed_value=33 + i,
                                 max_num_cities=20, grid_mode=True, max_rail_between_cities=2,
                                 max_rail_in_city=7, earliest_malfunction=15, malfunction_duration=15,
                                 speed_data={1: 1.0},
                                 number_of_shortest_paths_per_agent=10, weight_route_change=1,
                                 weight_lateness_seconds=1,
                                 max_window_size_from_earliest=np.inf)
            for i in range(nb_experiments)
        ]
    )
    # Run experiments
    experiment_folder_name = run_experiment_agenda(
        experiment_agenda=experiment_agenda,
        run_experiments_parallel=False,
        show_results_without_details=True,
        verbose=False)
    hypothesis_one_data_analysis(
        data_folder=experiment_folder_name,
        analysis_2d=True,
        analysis_3d=False,
        malfunction_analysis=False,
        qualitative_analysis_experiment_ids=range(nb_experiments))


if __name__ == '__main__':
    _run_experiment_ids_from_agenda()
