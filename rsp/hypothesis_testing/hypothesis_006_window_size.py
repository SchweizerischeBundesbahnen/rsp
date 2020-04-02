import numpy as np

from rsp.hypothesis_testing.run_null_alt_agenda import compare_agendas
from rsp.hypothesis_testing.tweak_experiment_agenda import tweak_max_window_size_from_earliest
from rsp.utils.experiments import load_experiment_agenda_from_file


def hypothesis_006_window_size_main(copy_agenda_from_base_directory: str):
    experiment_name = "plausi_006"
    agenda_null = tweak_max_window_size_from_earliest(
        agenda_null=load_experiment_agenda_from_file(copy_agenda_from_base_directory),
        max_window_size_from_earliest=np.inf,
        alt_index=None,
        experiment_name=experiment_name)(
        experiment_name=experiment_name)
    compare_agendas(
        experiment_name=experiment_name,
        experiment_agenda_null=agenda_null,
        experiment_agenda_alternatives=[
            tweak_max_window_size_from_earliest(
                agenda_null=agenda_null,
                max_window_size_from_earliest=s,
                alt_index=index,
                experiment_name=experiment_name)
            for index, s in enumerate([30, 60])],
        copy_agenda_from_base_directory=copy_agenda_from_base_directory
    )


if __name__ == '__main__':
    hypothesis_006_window_size_main(copy_agenda_from_base_directory='exp_hypothesis_one_2020_03_31T07_11_03')
