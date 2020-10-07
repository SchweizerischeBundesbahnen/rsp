"""Export."""
import argparse
import os
from shutil import copyfile
from typing import Callable
from typing import List

from importlib_resources import path

from rsp.experiment_solvers.asp.asp_problem_description import ASPProblemDescription
from rsp.experiment_solvers.data_types import SchedulingExperimentResult
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription
from rsp.utils.data_types import ExperimentResultsAnalysis
from rsp.utils.experiments import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.utils.experiments import EXPERIMENT_POTASSCO_SUBDIRECTORY_NAME
from rsp.utils.experiments import load_and_expand_experiment_results_from_data_folder
from rsp.utils.file_utils import check_create_folder


def potassco_export(experiment_potassco_directory: str,
                    experiment_results_list: List[ExperimentResultsAnalysis],
                    asp_export_experiment_ids: List[int],
                    export_schedule_full: bool = False,
                    export_reschedule_full_after_malfunction: bool = True,
                    export_reschedule_delta_perfect_after_malfunction: bool = False,
                    ):
    """Create subfolder potassco in the basefolder and export programs and data
    for the given experiment ids and shell script to start them.

    Parameters
    ----------
    experiment_potassco_directory
    experiment_results_list
    asp_export_experiment_ids
    export_schedule_full
    export_reschedule_full_after_malfunction
    export_reschedule_delta_perfect_after_malfunction
    """
    print(f"potassco export to {experiment_potassco_directory}")
    check_create_folder(experiment_potassco_directory)

    # filter
    filtered_experiments: List[ExperimentResultsAnalysis] = list(filter(
        lambda experiment: experiment.experiment_id in asp_export_experiment_ids,
        experiment_results_list))

    # write .lp and .sh
    schedule_programs = ["encoding.lp", "minimize_total_sum_of_running_times.lp"]
    reschedule_programs = ["encoding.lp", "delay_linear_within_one_minute.lp",
                           "minimize_delay_and_routes_combined.lp"]
    for experiment in filtered_experiments:
        experiment_id = experiment.experiment_id
        if export_schedule_full:
            _potassco_write_lp_and_sh_for_experiment(
                experiment_id=experiment_id,
                experiment_potassco_directory=experiment_potassco_directory,
                name="schedule_full",
                problem=experiment.problem_full,
                programs=[f"encoding/{s}" for s in schedule_programs],
                results=experiment.results_full,
                factory_method=ASPProblemDescription.factory_scheduling
            )
        if export_reschedule_full_after_malfunction:
            _potassco_write_lp_and_sh_for_experiment(
                experiment_id=experiment_id,
                experiment_potassco_directory=experiment_potassco_directory,
                name="reschedule_full_after_malfunction",
                problem=experiment.problem_full_after_malfunction,
                programs=[f"encoding/{s}" for s in reschedule_programs],
                results=experiment.results_full_after_malfunction,
                factory_method=ASPProblemDescription.factory_rescheduling
            )
        if export_reschedule_delta_perfect_after_malfunction:
            _potassco_write_lp_and_sh_for_experiment(
                experiment_id=experiment_id,
                experiment_potassco_directory=experiment_potassco_directory,
                name="reschedule_delta_perfect_after_malfunction",
                problem=experiment.problem_delta_perfect_after_malfunction,
                programs=[f"encoding/{s}" for s in reschedule_programs],
                results=experiment.results_delta_perfect_after_malfunction,
                factory_method=ASPProblemDescription.factory_rescheduling
            )

    # copy program files
    check_create_folder(f"{experiment_potassco_directory}/encoding")

    for file in schedule_programs + reschedule_programs:
        with path('res.asp.encodings', file) as src:
            copyfile(src, f"{experiment_potassco_directory}/encoding/{file}")

    # copy 2d analysis files for export to potassco
    analysis_folder = f"{experiment_potassco_directory}/../analysis"
    for subfolder in ["asp_plausi", "main_results"]:
        _copy_pdfs_from_analysis_subfolder_to_potassco_directory(
            analysis_folder=analysis_folder,
            experiment_potassco_directory=experiment_potassco_directory,
            subfolder=subfolder)


def _copy_pdfs_from_analysis_subfolder_to_potassco_directory(analysis_folder, experiment_potassco_directory, subfolder):
    asp_plausi_dest_folder = f"{experiment_potassco_directory}/{subfolder}"
    check_create_folder(asp_plausi_dest_folder)
    asp_plausi_src_folder = f"{analysis_folder}/{subfolder}"
    check_create_folder(asp_plausi_src_folder)
    log_files = os.listdir(asp_plausi_src_folder)
    for file in [file for file in log_files if file.endswith(".pdf")]:
        copyfile(f"{asp_plausi_src_folder}/{file}", f"{asp_plausi_dest_folder}/{file}")


def _potassco_write_lp_and_sh_for_experiment(
        experiment_id: int,
        experiment_potassco_directory: str, name: str,
        problem: ScheduleProblemDescription,
        programs: List[str],
        results: SchedulingExperimentResult,
        factory_method: Callable[[ScheduleProblemDescription, int], ASPProblemDescription]
):
    """Write .lp and .sh to the potassco folder.

    Parameters
    ----------
    experiment_id
    experiment_potassco_directory
    name
    problem
    programs
    results
    factory_method: Callable[[ScheduleProblemDescription, int], ASPProblemDescription]
        either `ASPProblemDescription.factory_scheduling` or `ASPProblemDescription.factory_rescheduling`
    """
    check_create_folder(experiment_potassco_directory)
    # TODO for cohesion, this should be part of asp_helper.py.
    #  However, we would have to refactor asp_helper.py too much in order to it.
    #  Do it later if this approach proves insufficient.
    file_name_prefix = f"{experiment_id :04d}_{name}"
    with open(f"{experiment_potassco_directory}/{file_name_prefix}.lp", "w") as out:
        solver_program = results.solver_program
        # temporary workaround: data from Erik were produced without the new filed solver_program
        if solver_program is None:
            asp_model = factory_method(problem, results.solver_seed)
            solver_program = asp_model.asp_program

        out.write("\n".join(solver_program))
    with open(f"{experiment_potassco_directory}/{file_name_prefix}.sh", "w", newline='\n') as out:
        out.write("clingo-dl " + " ".join(programs) +
                  f" {file_name_prefix}.lp "
                  f"--seed={results.solver_seed} "
                  f"-t2 --lookahead=no "
                  f"--propagate=partial "
                  f"--opt-mode=opt "
                  f"--models=0\n"
                  )
    with open(f"{experiment_potassco_directory}/{file_name_prefix}_statistics.txt", "w", newline='\n') as out:
        out.write(f"{results.solver_statistics}")
    with open(f"{experiment_potassco_directory}/{file_name_prefix}_configuration.txt", "w", newline='\n') as out:
        out.write(f"{results.solver_configuration}")
    with open(f"{experiment_potassco_directory}/{file_name_prefix}_result.txt", "w", newline='\n') as out:
        out.write(f"{results.solver_result}")


def main(experiment_base_directory: str,
         experiment_ids: List[int],
         problem: str):
    export_schedule_full: bool = False
    export_reschedule_full_after_malfunction: bool = False
    export_reschedule_delta_perfect_after_malfunction: bool = False
    if problem == "full":
        export_schedule_full = True
    elif problem == 'full_after_malfunction':
        export_reschedule_full_after_malfunction = True
    elif problem == 'delta_perfect_after_malfunction':
        export_reschedule_delta_perfect_after_malfunction = True
    else:
        raise ValueError(f"unkonwn problem={problem}")
    experiment_data_directory = f'{experiment_base_directory}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}'
    experiment_results_list = load_and_expand_experiment_results_from_data_folder(
        experiment_data_folder_name=experiment_data_directory, experiment_ids=experiment_ids)
    potassco_export(
        experiment_potassco_directory=f'{experiment_base_directory}/{EXPERIMENT_POTASSCO_SUBDIRECTORY_NAME}',
        experiment_results_list=experiment_results_list, asp_export_experiment_ids=experiment_ids,
        export_schedule_full=export_schedule_full,
        export_reschedule_full_after_malfunction=export_reschedule_full_after_malfunction,
        export_reschedule_delta_perfect_after_malfunction=export_reschedule_delta_perfect_after_malfunction
    )


if __name__ == '__main__':
    # sample call:
    # python rsp/asp_plausibility/potassco_export.py --experiment_base_directory=../rsp-data/agent_0_malfunction_2020_05_27T19_45_49
    #   --experiment_id=0 --problem=full_after_malfunction
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_base_directory', type=str, nargs=1, help='../rsp-data/agent_0_malfunction_2020_05_27T19_45_49')
    parser.add_argument('--experiment_id', type=int, nargs=1, help='0,1,2,3...')
    parser.add_argument('--problem', type=str,
                        choices=['full_after_malfunction', 'full', 'delta_perfect_after_malfunction'],
                        help='which problem to check',
                        nargs=1)
    args = parser.parse_args()
    main(experiment_base_directory=args.experiment_base_directory[0], experiment_ids=[args.experiment_id[0]], problem=args.problem[0])
