import os
from typing import List
from typing import Optional

from rsp.hypothesis_testing.run_null_alt_agenda import compare_agendas
from rsp.hypothesis_testing.tweak_experiment_agenda import merge_agendas_under_new_name
from rsp.hypothesis_testing.tweak_experiment_agenda import tweak_asp_seed_value
from rsp.utils.experiments import EXPERIMENT_AGENDA_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_experiment_agenda_from_file


def hypothesis_001_solver_seeding_irrelevant_main(
        copy_agenda_from_base_directory: str,
        experiment_ids: Optional[List[int]] = None
):
    """Copy agenda (A.1) and schedule/malfunction (A.2) and run pipeline from B
    multiple times with tweaked agenda and compare the runs D.

    Parameters
    ----------
    copy_agenda_from_base_directory: str
    experiment_ids: Optional[List[int]]
    """
    experiment_name = "plausi_001"
    agenda_null = load_experiment_agenda_from_file(
        f"{copy_agenda_from_base_directory}/{EXPERIMENT_AGENDA_SUBDIRECTORY_NAME}")
    compare_agendas(
        experiment_name=experiment_name,
        experiment_ids=experiment_ids,
        experiment_agenda=merge_agendas_under_new_name(experiment_name=experiment_name, agendas=[agenda_null] + [
            tweak_asp_seed_value(
                agenda_null=agenda_null, seed=(94 + inc),
                alt_index=inc,
                experiment_name=experiment_name)
            for inc in range(5)]),
        # TODO column and baseline value
        copy_agenda_from_base_directory=copy_agenda_from_base_directory,
        run_analysis=False,
        parallel_compute=True
    )


if __name__ == '__main__':
    hypothesis_001_solver_seeding_irrelevant_main(
        copy_agenda_from_base_directory=os.path.abspath('exp_hypothesis_one_2020_03_31T07_11_03')
    )
