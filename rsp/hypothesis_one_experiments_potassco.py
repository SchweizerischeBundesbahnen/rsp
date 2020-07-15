import re
from typing import Optional

from rsp.hypothesis_one_experiments import get_agenda_pipeline_params_003_a_bit_more_advanced
from rsp.hypothesis_one_experiments import hypothesis_one_pipeline
from rsp.hypothesis_one_experiments import hypothesis_one_rerun_without_regen_schedule


# TODO pass arguments instead of hacky file editing
def enable_seq(enable=True):
    off = "RESCHEDULE_HEURISTICS = []"
    on = "RESCHEDULE_HEURISTICS = [ASPHeuristics.HEURISTIC_SEQ]"
    file_name = "rsp/utils/global_constants.py"
    with open(file_name, "r") as fh:
        output_str = fh.read().replace(off if enable else on, on if enable else off)
    with open(file_name, "w") as output:
        output.write(output_str)


# TODO pass arguments instead of hacky file editing
def set_delay_model_resolution(resolution=1):
    file_name = "rsp/utils/global_constants.py"
    with open(file_name, "r") as fh:
        regex = re.compile("DELAY_MODEL_RESOLUTION = .*")
        output_str = regex.sub(f"DELAY_MODEL_RESOLUTION = {resolution}", fh.read())
    with open(file_name, "w") as output:
        output.write(output_str)


# TODO pass arguments instead of hacky file editing
def enable_propagate_partial(enable: bool = True):
    file_name = "rsp/utils/global_constants.py"
    with open(file_name, "r") as fh:
        regex = re.compile("DL_PROPAGATE_PARTIAL = .*")
        output_str = regex.sub(f"DL_PROPAGATE_PARTIAL = True" if enable else f"DL_PROPAGATE_PARTIAL = False", fh.read())
    with open(file_name, "w") as output:
        output.write(output_str)


def set_defaults():
    enable_seq(False)
    set_delay_model_resolution(1)
    enable_propagate_partial(enable=True)


def main(gen_schedule: bool = True, run_experiments: bool = True, copy_agenda_from_base_directory: Optional[str] = None):
    """

    Parameters
    ----------
    gen_schedule
        generate schedule? If `False`, `copy_agenda_from_base_directory` must be provided.
    run_experiments
        run experiments after schedule generation
    copy_agenda_from_base_directory
    """
    if gen_schedule:
        copy_agenda_from_base_directory = hypothesis_one_pipeline(
            experiment_name='003_a_bit_more_advanced_schedules_only',
            parameter_ranges_and_speed_data=get_agenda_pipeline_params_003_a_bit_more_advanced(),
            qualitative_analysis_experiment_ids=[],
            asp_export_experiment_ids=[],
            experiment_ids=None,
            gen_only=True
        )
    if run_experiments and copy_agenda_from_base_directory is not None:
        try:
            nb_runs = 3
            experiment_name_prefix = copy_agenda_from_base_directory + "_"
            parallel_compute = 2
            experiment_ids = None
            # baseline with defaults
            set_defaults()
            hypothesis_one_rerun_without_regen_schedule(
                copy_agenda_from_base_directory=copy_agenda_from_base_directory,
                experiment_name=('%sbaseline' % experiment_name_prefix),
                nb_runs=nb_runs,
                parallel_compute=parallel_compute,
                experiment_ids=experiment_ids,
                run_analysis=False,
                with_file_handler_to_rsp_logger=True
            )
            # effect of SEQ heuristic (SIM-167)
            set_defaults()
            enable_seq(True)
            hypothesis_one_rerun_without_regen_schedule(
                copy_agenda_from_base_directory=copy_agenda_from_base_directory,
                experiment_name=('%swith_SEQ' % experiment_name_prefix),
                nb_runs=nb_runs,
                parallel_compute=parallel_compute,
                experiment_ids=experiment_ids,
                run_analysis=False,
                with_file_handler_to_rsp_logger=True
            )
            # effect of delay model resolution (SIM-542)
            set_defaults()
            set_delay_model_resolution(2)
            hypothesis_one_rerun_without_regen_schedule(
                copy_agenda_from_base_directory=copy_agenda_from_base_directory,
                experiment_name=('%swith_delay_model_resolution_2' % experiment_name_prefix),
                nb_runs=nb_runs,
                parallel_compute=parallel_compute,
                experiment_ids=experiment_ids,
                run_analysis=False,
                with_file_handler_to_rsp_logger=True
            )
            set_defaults()
            set_delay_model_resolution(5)
            hypothesis_one_rerun_without_regen_schedule(
                copy_agenda_from_base_directory=copy_agenda_from_base_directory,
                experiment_name=('%swith_delay_model_resolution_5' % experiment_name_prefix),
                nb_runs=nb_runs,
                parallel_compute=parallel_compute,
                experiment_ids=experiment_ids,
                run_analysis=False,
                with_file_handler_to_rsp_logger=True
            )
            set_defaults()
            set_delay_model_resolution(10)
            hypothesis_one_rerun_without_regen_schedule(
                copy_agenda_from_base_directory=copy_agenda_from_base_directory,
                experiment_name=('%swith_delay_model_resolution_10' % experiment_name_prefix),
                nb_runs=nb_runs,
                parallel_compute=parallel_compute,
                experiment_ids=experiment_ids,
                run_analysis=False,
                with_file_handler_to_rsp_logger=True
            )
            # # effect of --propagate (SIM-543)
            set_defaults()
            enable_propagate_partial(enable=False)
            hypothesis_one_rerun_without_regen_schedule(
                copy_agenda_from_base_directory=copy_agenda_from_base_directory,
                experiment_name=('%swithout_propagate_partial' % experiment_name_prefix),
                nb_runs=nb_runs,
                parallel_compute=parallel_compute,
                experiment_ids=experiment_ids,
                run_analysis=False,
                with_file_handler_to_rsp_logger=True
            )
        finally:
            set_defaults()


if __name__ == '__main__':
    main(
        gen_schedule=False,
        run_experiments=True,
        copy_agenda_from_base_directory='../rsp-data/003_a_bit_more_advanced_schedules_only_2020_06_12T21_01_45_merge_2020_06_19T16_23_16'
    )
