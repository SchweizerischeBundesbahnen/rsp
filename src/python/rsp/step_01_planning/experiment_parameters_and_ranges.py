from typing import List
from typing import Mapping
from typing import NamedTuple

from rsp.utils.global_constants import GlobalConstants

SpeedData = Mapping[float, float]

# @deprecated(reason="You should use hiearchy level ranges.")
ParameterRanges = NamedTuple(
    "ParameterRanges",
    [
        # infrastructure and agent placement
        # 0: size_range
        ("size_range", List[int]),
        # 1: agent_range
        ("agent_range", List[int]),
        # 2: in_city_rail_range
        ("in_city_rail_range", List[int]),
        # 3: out_city_rail_range
        ("out_city_rail_range", List[int]),
        # 4: city_range
        ("city_range", List[int]),
        # malfunction
        # 5: earliest_malfunction
        ("earliest_malfunction", List[int]),
        # 6: malfunction_duration
        ("malfunction_duration", List[int]),
        # rescheduling
        # 7: number_of_shortest_paths_per_agent
        ("number_of_shortest_paths_per_agent", List[int]),
        # 8: max_window_size_from_earliest
        ("max_window_size_from_earliest", List[int]),
        # 9: asp_seed_value
        ("asp_seed_value", List[int]),
        # 10: weight_route_change
        ("weight_route_change", List[int]),
        # 11: weight_lateness_seconds
        ("weight_lateness_seconds", List[int]),
    ],
)

# TODO remove deprecated ParameterRangesAndSpeedData
# @deprecated(reason="You should use hiearchy level ranges.") # deprecated decorator only applies to functions, therefore comment out
ParameterRangesAndSpeedData = NamedTuple("ParameterRangesAndSpeedData", [("parameter_ranges", ParameterRanges), ("speed_data", SpeedData)])

InfrastructureParameters = NamedTuple(
    "InfrastructureParameters",
    [
        ("infra_id", int),
        ("width", int),
        ("height", int),
        ("flatland_seed_value", int),
        ("max_num_cities", int),
        ("grid_mode", bool),
        ("max_rail_between_cities", int),
        ("max_rail_in_city", int),
        ("number_of_agents", int),
        ("speed_data", SpeedData),
        ("number_of_shortest_paths_per_agent", int),
    ],
)

InfrastructureParametersRange = NamedTuple(
    "InfrastructureParameters",
    [
        ("width", List[int]),
        ("height", List[int]),
        ("flatland_seed_value", List[int]),
        ("max_num_cities", List[int]),
        ("max_rail_between_cities", List[int]),
        ("max_rail_in_city", List[int]),
        ("number_of_agents", List[int]),
        ("number_of_shortest_paths_per_agent", List[int]),
    ],
)

ScheduleParameters = NamedTuple(
    "ScheduleParameters", [("infra_id", int), ("schedule_id", int), ("asp_seed_value", int), ("number_of_shortest_paths_per_agent_schedule", int)]
)

ScheduleParametersRange = NamedTuple("ScheduleParametersRange", [("asp_seed_value", List[int]), ("number_of_shortest_paths_per_agent_schedule", List[int])])

ReScheduleParametersRange = NamedTuple(
    "ReScheduleParametersRange",
    [
        # 5: earliest_malfunction
        ("earliest_malfunction", List[int]),
        # 6: malfunction_duration
        ("malfunction_duration", List[int]),
        # 7:
        ("malfunction_agent_id", List[int]),
        # rescheduling
        # 7: number_of_shortest_paths_per_agent
        ("number_of_shortest_paths_per_agent", List[int]),
        # 8: max_window_size_from_earliest
        ("max_window_size_from_earliest", List[int]),
        # 9: asp_seed_value
        ("asp_seed_value", List[int]),
        # 10: weight_route_change
        ("weight_route_change", List[int]),
        # 11: weight_lateness_seconds
        ("weight_lateness_seconds", List[int]),
    ],
)

ReScheduleParameters = NamedTuple(
    "ReScheduleParameters",
    [
        # 5: earliest_malfunction
        ("earliest_malfunction", int),
        # 6: malfunction_duration
        ("malfunction_duration", int),
        ("malfunction_agent_id", int),
        # rescheduling
        # 7: number_of_shortest_paths_per_agent
        ("number_of_shortest_paths_per_agent", int),
        # 8: max_window_size_from_earliest
        ("max_window_size_from_earliest", int),
        # 9: asp_seed_value
        ("asp_seed_value", int),
        # 10: weight_route_change
        ("weight_route_change", int),
        # 11: weight_lateness_seconds
        ("weight_lateness_seconds", int),
    ],
)

# the experiment_id is unambiguous within the agenda for the full parameter set!
ExperimentParameters = NamedTuple(
    "ExperimentParameters",
    [
        ("experiment_id", int),  # unique per execution (there may be multiple `experiment_id`s for the same `grid_id`
        ("grid_id", int),  # same if all params are the same
        ("infra_id_schedule_id", int),
        ("infra_parameters", InfrastructureParameters),
        ("schedule_parameters", ScheduleParameters),
        ("re_schedule_parameters", ReScheduleParameters),
    ],
)

ExperimentAgenda = NamedTuple(
    "ExperimentAgenda", [("experiment_name", str), ("experiments", List[ExperimentParameters]), ("global_constants", GlobalConstants)]
)
