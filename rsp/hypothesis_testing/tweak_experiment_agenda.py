from typing import List
from typing import Optional

from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentParameters


def tweak_name(
        agenda_null: ExperimentAgenda,
        alt_index: Optional[int],
        experiment_name: str) -> ExperimentAgenda:
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
    suffix = _make_suffix(alt_index)
    return ExperimentAgenda(
        experiment_name=f"{experiment_name}_{suffix}",
        experiments=[
            ExperimentParameters(**dict(experiment._asdict(),
                                        **{'max_window_size_from_earliest': max_window_size_from_earliest}))
            for experiment in agenda_null.experiments]
    )


def merge_agendas_under_new_name(experiment_name: str, agendas: List[ExperimentAgenda]) -> ExperimentAgenda:
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
