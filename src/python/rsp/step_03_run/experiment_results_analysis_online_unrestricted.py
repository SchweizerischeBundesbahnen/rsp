"""`ExperimentResultsAnalysis` contains data structures for analysis for scope
`online_unrestricted` only, with/without raw `ExperimentResults`.

Data structure should be mostly be flat (fields should be numbers) and
only some agent dicts that are often re-used.
"""
from typing import List
from typing import NamedTuple

from pandas import DataFrame
from rsp.step_03_run.experiment_results import ExperimentResults
from rsp.step_03_run.experiment_results_analysis import experiment_results_analysis_all_scopes_fields
from rsp.step_03_run.experiment_results_analysis import extract_all_scopes_fields
from rsp.step_03_run.experiment_results_analysis import extract_base_fields

ExperimentResultsAnalysisOnlineUnrestricted = NamedTuple(
    "ExperimentResultsAnalysisOnlineUnrestricted",
    [
        ("experiment_id", int),
        ("grid_id", int),
        ("infra_id", int),
        ("schedule_id", int),
        ("infra_id_schedule_id", int),
        ("size", int),
        ("n_agents", int),
        ("max_num_cities", int),
        ("max_rail_between_cities", int),
        ("max_rail_in_city", int),
        ("earliest_malfunction", int),
        ("malfunction_duration", int),
        ("malfunction_agent_id", int),
        ("weight_route_change", int),
        ("weight_lateness_seconds", int),
        ("max_window_size_from_earliest", int),
        ("n_agents_running", int),
        ("rescheduling_horizon", int),
    ]
    + [(f"{prefix}_{scope}", type_) for prefix, (type_, _) in experiment_results_analysis_all_scopes_fields.items() for scope in ["online_unrestricted"]],
)


# TODO SIM-749 check notebooks again!
# TODO SIM-749 mark csv as containing only online unrestricted?!
def expand_experiment_results_online_unrestricted(experiment_results: ExperimentResults) -> ExperimentResultsAnalysisOnlineUnrestricted:
    return ExperimentResultsAnalysisOnlineUnrestricted(
        **extract_base_fields(
            experiment_parameters=experiment_results.experiment_parameters,
            problem_online_unrestricted=experiment_results.problem_online_unrestricted,
            malfunction=experiment_results.malfunction,
            results_schedule=experiment_results.results_schedule,
        ),
        **extract_all_scopes_fields(experiment_results=experiment_results, all_scopes=["online_unrestricted"]),
    )


def convert_list_of_experiment_results_analysis_online_unrestricted_to_data_frame(l: List[ExperimentResultsAnalysisOnlineUnrestricted]) -> DataFrame:
    df = DataFrame(columns=ExperimentResultsAnalysisOnlineUnrestricted._fields, data=[r._asdict() for r in l])
    df = df.select_dtypes(exclude=["object"])
    return df
