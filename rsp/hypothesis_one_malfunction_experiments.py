import os
from typing import Dict

from rsp.hypothesis_one_pipeline_all_in_one import hypothesis_one_gen_schedule
from rsp.hypothesis_testing.utils.tweak_experiment_agenda import tweak_parameter_ranges
from rsp.logger import rsp_logger
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import ParameterRangesAndSpeedData
from rsp.utils.experiments import load_parameter_ranges_and_speed_data
from rsp.utils.experiments import run_experiment_agenda


def get_agenda_pipeline_malfunction_variation() -> ParameterRangesAndSpeedData:
    parameter_ranges = ParameterRanges(agent_range=[50, 50, 1],
                                       size_range=[75, 75, 1],
                                       in_city_rail_range=[3, 3, 1],
                                       out_city_rail_range=[2, 2, 1],
                                       city_range=[5, 5, 1],
                                       earliest_malfunction=[1, 1, 1],
                                       malfunction_duration=[50, 50, 1],
                                       number_of_shortest_paths_per_agent=[10, 10, 1],
                                       max_window_size_from_earliest=[100, 100, 1],
                                       asp_seed_value=[1, 1, 1],
                                       # route change is penalized the same as 1 second delay
                                       weight_route_change=[20, 20, 1],
                                       weight_lateness_seconds=[1, 1, 1]
                                       )

    # Define the desired speed profiles
    speed_data = {1.: 0.25,  # Fast passenger train
                  1. / 2.: 0.25,  # Fast freight train
                  1. / 3.: 0.25,  # Slow commuter train
                  1. / 4.: 0.25}  # Slow freight train
    return ParameterRangesAndSpeedData(parameter_ranges=parameter_ranges, speed_data=speed_data)


# TODO SIM-650 should work again
def hypothesis_one_malfunction_analysis(
        base_directory: str,
        gen_schedule: bool = False,
        experiment_name: str = None,
        base_experiment_id: int = 0,
        malfunction_agent_id: int = 0,
        parameter_ranges_and_speed_data: ParameterRangesAndSpeedData = None,
        malfunction_ranges: Dict = None,
        flatland_seed: int = 12,
        parallel_compute: int = 5, EXPERIMENT_AGENDA_SUBDIRECTORY_NAME=None):
    rsp_logger.info(f"MALFUNCTION INVESTIGATION")

    # Generate Schedule
    if gen_schedule:
        experiment_base_folder_name = hypothesis_one_gen_schedule(parameter_ranges_and_speed_data,
                                                                  experiment_name=experiment_name,
                                                                  flatland_seed=flatland_seed)
    # Use existing Schedule
    else:
        experiment_agenda_directory = f'{base_directory}/{EXPERIMENT_AGENDA_SUBDIRECTORY_NAME}'
        parameter_ranges_and_speed_data: ParameterRangesAndSpeedData = load_parameter_ranges_and_speed_data(experiment_folder_name=experiment_agenda_directory)
        if parameter_ranges_and_speed_data is None:
            rsp_logger.info("No parameters found. Reverting to default!")
            parameter_ranges_and_speed_data: ParameterRangesAndSpeedData = get_agenda_pipeline_malfunction_variation()

        experiment_base_folder_name = base_directory

    # Update the loaded or provided parameters with the new malfunction parameters
    parameter_ranges_and_speed_data = tweak_parameter_ranges(original_ranges_and_data=parameter_ranges_and_speed_data, new_parameter_ranges=malfunction_ranges)

    # TODO SIM-650 should work again -> integration test
    run_experiment_agenda(
        copy_agenda_from_base_directory=experiment_base_folder_name,
        parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
        base_experiment_id=base_experiment_id,
        experiment_agenda_name=f"agent_{malfunction_agent_id}_malfunction",
        parallel_compute=parallel_compute,
        malfunction_agent_id=malfunction_agent_id
    )


if __name__ == '__main__':
    # do not commit your own calls !
    # Define an experiment name, if the experiment already exists we load the schedule from existing experiment
    # Beware of time-stamps when re-runing experiments

    # Generate schedule with n_agents
    n_agents = 105
    experiment_name = 'schedule_{}_agents'.format(n_agents)
    experiment_base_directory = 'None'
    if not os.path.exists(experiment_base_directory):
        experiment_base_directory = None

    # Define parameters for experiment, these are only used if the experiment is not loaded from old data
    parameter_ranges_and_speed_data: ParameterRangesAndSpeedData = get_agenda_pipeline_malfunction_variation()

    # Update n_agent ranges and see,
    if experiment_base_directory is None:
        new_agent_ranges = {'agent_range': [n_agents, n_agents, 1],
                            'size_range': [100, 100, 1],
                            'in_city_rail_range': [2, 2, 1],
                            'out_city_rail_range': [1, 1, 1],
                            'city_range': [20, 20, 1]}
        parameter_ranges_and_speed_data = tweak_parameter_ranges(original_ranges_and_data=parameter_ranges_and_speed_data,
                                                                 new_parameter_ranges=new_agent_ranges)

    # Vary the malfunction
    malfunction_ranges = {'earliest_malfunction': [1, 300, 50],
                          'malfunction_duration': [50, 50, 1]}
    malfunction_agent_id = 34

    # Run the malfunction variation experiments
    hypothesis_one_malfunction_analysis(base_directory=experiment_base_directory,
                                        experiment_name=experiment_name,
                                        parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
                                        malfunction_ranges=malfunction_ranges,
                                        malfunction_agent_id=malfunction_agent_id,
                                        flatland_seed=14)
