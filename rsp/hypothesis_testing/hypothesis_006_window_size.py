"""Plausibility hypothesis 006:
https://confluence.sbb.ch/display/SIM/006_constant_time_window."""
import os
from typing import List
from typing import Optional

import numpy as np

from rsp.hypothesis_testing.utils.run_null_alt_agenda import compare_agendas
from rsp.hypothesis_testing.utils.tweak_experiment_agenda import merge_agendas_under_new_name
from rsp.hypothesis_testing.utils.tweak_experiment_agenda import tweak_max_window_size_from_earliest
from rsp.utils.experiments import EXPERIMENT_AGENDA_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_experiment_agenda_from_file


def hypothesis_006_window_size_main(
        copy_agenda_from_base_directory: str,
        experiment_ids: Optional[List[int]] = None):
    experiment_name = "plausi_006"
    agenda_null = tweak_max_window_size_from_earliest(
        agenda_null=load_experiment_agenda_from_file(
            f"{copy_agenda_from_base_directory}/{EXPERIMENT_AGENDA_SUBDIRECTORY_NAME}"),
        max_window_size_from_earliest=np.inf,
        alt_index=None,
        experiment_name=experiment_name)
    compare_agendas(
        experiment_name=experiment_name,
        experiment_ids=experiment_ids,
        experiment_agenda=merge_agendas_under_new_name(experiment_name=experiment_name, agendas=[agenda_null] + [
            tweak_max_window_size_from_earliest(
                agenda_null=agenda_null,
                max_window_size_from_earliest=s,
                alt_index=index,
                experiment_name=experiment_name)
            for index, s in enumerate([30, 60])]),
        # TODO column and baseline value
        copy_agenda_from_base_directory=copy_agenda_from_base_directory,
        run_analysis=False,
        parallel_compute=True
    )


if __name__ == '__main__':
    hypothesis_006_window_size_main(
        copy_agenda_from_base_directory=os.path.abspath('exp_hypothesis_one_2020_03_31T07_11_03')
    )
