import os

from rsp.hypothesis_one_experiments import get_agenda_pipeline_malfunction_variation
from rsp.hypothesis_one_experiments import hypothesis_one_malfunction_analysis
from rsp.utils.data_types import ParameterRangesAndSpeedData

if __name__ == '__main__':
    # do not commit your own calls !
    # Define an experiment name, if the experiment already exists we load the schedule from existing experiment
    # Beware of time-stamps when re-runing experiments

    experiment_name = 'schedule_120_agents'
    experiment_base_directory = '../rsp-data/{}/'.format(experiment_name)
    if not os.path.exists(experiment_base_directory):
        experiment_base_directory = None

    # Define parameters for experiment, these are only used if the experiment is not loaded from old data
    parameter_ranges_and_speed_data: ParameterRangesAndSpeedData = get_agenda_pipeline_malfunction_variation()

    # Vary the malfunction
    malfunction_ranges = {'earliest_malfunction': [1, 300, 30],
                          'malfunction_duration': [50, 50, 1]}
    malfunction_agent_id = 5

    # Run the malfunction variation experiments
    hypothesis_one_malfunction_analysis(agenda_folder=experiment_base_directory,
                                        experiment_name=experiment_name,
                                        parameter_ranges_and_speed_data=parameter_ranges_and_speed_data,
                                        malfunction_ranges=malfunction_ranges,
                                        malfunction_agent_id=malfunction_agent_id)
