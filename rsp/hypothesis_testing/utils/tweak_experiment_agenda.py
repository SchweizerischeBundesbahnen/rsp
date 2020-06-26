from typing import Dict
from typing import List
from typing import Optional

from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ParameterRanges
from rsp.utils.data_types import ParameterRangesAndSpeedData


def tweak_name(
        agenda_null: ExperimentAgenda,
        alt_index: Optional[int],
        experiment_name: str) -> ExperimentAgenda:
    """Produce a new `ExperimentAgenda` under a "tweaked" name.

    Parameters
    ----------
    agenda_null
    alt_index
    experiment_name

    Returns
    -------
    """
    suffix = _make_suffix(alt_index)
    return ExperimentAgenda(
        experiment_name=f"{experiment_name}_{suffix}",
        experiments=agenda_null.experiments
    )


def tweak_asp_seed_value(
        agenda_null: ExperimentAgenda,
        seed: int,
        alt_index: Optional[int],
        experiment_name: str,
) -> ExperimentAgenda:
    """Produce a new `ExperimentAgenda` with `asp_seed_value` "tweaked".

    Parameters
    ----------
    agenda_null
    seed
    alt_index
    experiment_name

    Returns
    -------
    """
    suffix = _make_suffix(alt_index)
    return ExperimentAgenda(
        experiment_name=f"{experiment_name}_{suffix}",
        experiments=[
            ExperimentParameters(**dict(experiment._asdict(),
                                        **{'asp_seed_value': seed}))
            for experiment in agenda_null.experiments]
    )


def tweak_max_window_size_from_earliest(
        agenda_null: ExperimentAgenda,
        max_window_size_from_earliest: int,
        alt_index: Optional[int],
        experiment_name: str) -> ExperimentAgenda:
    """Produce a new `ExperimentAgenda` with `max_window_size_from_earliest`
    "tweaked".

    Parameters
    ----------
    agenda_null
    max_window_size_from_earliest
    alt_index
    experiment_name

    Returns
    -------
    """
    suffix = _make_suffix(alt_index)
    return ExperimentAgenda(
        experiment_name=f"{experiment_name}_{suffix}",
        experiments=[
            ExperimentParameters(**dict(experiment._asdict(),
                                        **{'max_window_size_from_earliest': max_window_size_from_earliest}))
            for experiment in agenda_null.experiments]
    )


def tweak_experiment_agenda_parameters(
        agenda_null: ExperimentAgenda,
        new_parameters: Dict,
        suffix: Optional[str],
        experiment_name: str) -> ExperimentAgenda:
    """Produce a new `ExperimentAgenda` with tweaked parameter "new_parameters"
    This returns an agenda with the same experiments but changed parameters.
    The changes are applied to all experiment in the agenda.

    Parameters
    ----------
    agenda_null
        initial agenda we want to tweak
    new_parameters
        Dict of parameters that are changed
    suffix
        Suffix used to identify what changed
    experiment_name
        Name of experiment

    Returns
    -------
    """
    return ExperimentAgenda(
        experiment_name=f"{experiment_name}_{suffix}",
        experiments=[
            ExperimentParameters(**dict(experiment._asdict(),
                                        **new_parameters))
            for experiment in agenda_null.experiments]
    )


def tweak_parameter_ranges(
        original_ranges_and_data: ParameterRangesAndSpeedData,
        new_parameter_ranges: Dict) -> ParameterRangesAndSpeedData:
    """Change parameter ranges and speed data. Takes an original
    ParameterRangesAndSpeedData and updates the ranges specified in the
    new_parameter_ranges dict.

    Parameters
    ----------
    original_ranges_and_data
        Inital ranges and speed data
    new_parameters
        Dict of parameters that are changed


    Returns
    -------
    """
    ranges_dict = original_ranges_and_data.parameter_ranges._asdict()
    new_ranges = ParameterRanges(**dict(ranges_dict, **new_parameter_ranges))
    return ParameterRangesAndSpeedData(
        parameter_ranges=new_ranges,
        speed_data=original_ranges_and_data.speed_data
    )


def merge_agendas_under_new_name(experiment_name: str, agendas: List[ExperimentAgenda]) -> ExperimentAgenda:
    """Merge two agendas under a new name. Notice that `experiment_id`s may
    overlap and the merged agenda may have duplicate ids. This side-effect is
    exploited in `compare_agendas`.

    Parameters
    ----------
    experiment_name
    agendas

    Returns
    -------
    """
    return ExperimentAgenda(experiment_name=experiment_name, experiments=[
        experiment
        for experiment_agenda in agendas
        for experiment in experiment_agenda.experiments
    ])


def _make_suffix(alt_index: Optional[int]) -> str:
    """Make suffix for experiment name: either "null" if `alt_index` is `None`,
    else `alt{alt_index:03d}`

    Parameters
    ----------
    alt_index

    Returns
    -------
    """
    suffix = "null"
    if alt_index is not None:
        suffix = f"alt{alt_index:03d}"
    return suffix
