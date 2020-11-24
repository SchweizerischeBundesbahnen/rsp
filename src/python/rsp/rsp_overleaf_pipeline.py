from rsp.pipeline.rsp_pipeline_baseline_and_calibrations import rsp_pipeline_baseline_and_calibrations
from rsp.step_01_planning.experiment_parameters_and_ranges import ExperimentParameters
from rsp.step_01_planning.experiment_parameters_and_ranges import InfrastructureParametersRange
from rsp.step_01_planning.experiment_parameters_and_ranges import ReScheduleParametersRange
from rsp.step_01_planning.experiment_parameters_and_ranges import ScheduleParametersRange

INFRA_PARAMETERS_RANGE = InfrastructureParametersRange(
    number_of_agents=[50, 100, 4],
    width=[100, 100, 1],
    height=[100, 100, 1],
    flatland_seed_value=[190, 190, 1],
    max_num_cities=[8, 15, 3],
    max_rail_in_city=[2, 3, 2],
    max_rail_between_cities=[1, 2, 2],
    number_of_shortest_paths_per_agent=[10, 10, 1],
)
SCHEDULE_PARAMETERS_RANGE = ScheduleParametersRange(asp_seed_value=[814, 814, 1], number_of_shortest_paths_per_agent_schedule=[1, 1, 1],)
RESCHEDULE_PARAMETERS_RANGE = ReScheduleParametersRange(
    earliest_malfunction=[30, 30, 1],
    malfunction_duration=[50, 50, 1],
    # take all agents (200 is larger than largest number of agents)
    malfunction_agent_id=[0, 200, 200],
    number_of_shortest_paths_per_agent=[10, 10, 1],
    max_window_size_from_earliest=[60, 60, 1],
    asp_seed_value=[99, 99, 1],
    # route change is penalized the same as 30 seconds delay
    weight_route_change=[30, 30, 1],
    weight_lateness_seconds=[1, 1, 1],
)


def experiment_filter_first_ten_of_each_schedule(experiment: ExperimentParameters):
    return experiment.re_schedule_parameters.malfunction_agent_id < 100


if __name__ == "__main__":
    rsp_pipeline_baseline_and_calibrations(
        infra_parameters_range=INFRA_PARAMETERS_RANGE,
        schedule_parameters_range=SCHEDULE_PARAMETERS_RANGE,
        reschedule_parameters_range=RESCHEDULE_PARAMETERS_RANGE,
        base_directory="PUBLICATION_DATA",
        experiment_output_base_directory=None,
        experiment_filter=experiment_filter_first_ten_of_each_schedule,
        grid_mode=False,
        speed_data={
            1.0: 0.25,  # Fast passenger train
            1.0 / 2.0: 0.25,  # Fast freight train
            1.0 / 3.0: 0.25,  # Slow commuter train
            1.0 / 4.0: 0.25,  # Slow freight train
        },
    )
