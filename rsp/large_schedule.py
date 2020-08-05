import os

from rsp.hypothesis_one_experiments import get_agenda_pipeline_malfunction_variation
from rsp.hypothesis_one_experiments import hypothesis_one_gen_schedule
from rsp.hypothesis_testing.utils.tweak_experiment_agenda import tweak_parameter_ranges
from rsp.utils.data_types import ParameterRangesAndSpeedData

if __name__ == '__main__':
    # do not commit your own calls !
    # Define an experiment name, if the experiment already exists we load the schedule from existing experiment
    # Beware of time-stamps when re-runing experiments

    # Generate schedule with n_agents
    n_agents = 80
    experiment_name = 'large_schedule_{}_agents'.format(n_agents)
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

    # Run the malfunction variation experiments
    hypothesis_one_gen_schedule(
        experiment_name=experiment_name,
        parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
        flatland_seed=15)
