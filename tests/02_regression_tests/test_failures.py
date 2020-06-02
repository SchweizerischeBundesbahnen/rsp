from rsp.experiment_solvers.experiment_solver import ASPExperimentSolver
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.experiments import create_schedule_and_malfunction
from rsp.utils.experiments import load_experiment_agenda_from_file
from rsp.utils.experiments import load_schedule_and_malfunction
from rsp.utils.experiments import run_experiment_from_schedule_and_malfunction
from rsp.utils.experiments import save_schedule_and_malfunction


def test_exp_006_hypothesis_window_size_null_2020_03_17T18_34_47_experiment_158(
        verbose=True,
        debug=True,
        recreate_pickle=False
):
    """Test-driven and regression test for SIM-355: STOP agent at malfunction
    time is ignored by FLATland.

    We pass the entry times from ASP to FLATland's `ControllerFromTrainruns`,
    which derives when to take which FLATland actions for which agents.
    However, this `ControllerFromTrainruns` does not know about malfunctions.
    This regression test tests our fix (currently in RSP `create_controller_from_trainruns_and_malfunction`,
    should be moved to FLATland in the future):
    if STOP at malfunction start and no action at malfunction end, then issue the STOP at malfunction end.
    """
    experiment_agenda_directory = "tests/02_regression_tests/data/exp_006_hypothesis_window_size_null_2020_03_17T18_34_47/agenda"
    experiment_agenda = load_experiment_agenda_from_file(experiment_agenda_directory)
    schedule_and_malfunction = load_schedule_and_malfunction(
        experiment_agenda_directory=experiment_agenda_directory,
        experiment_id=158)
    experiment_parameters: ExperimentParameters = experiment_agenda.experiments[158]

    # SIM-517 dirty hack, re-create to remove dummy stuff
    if recreate_pickle:
        schedule_and_malfunction = create_schedule_and_malfunction(
            experiment_parameters=experiment_parameters,
            solver=ASPExperimentSolver())
        save_schedule_and_malfunction(schedule_and_malfunction=schedule_and_malfunction,
                                      experiment_agenda_directory=experiment_agenda_directory,
                                      experiment_id=experiment_parameters.experiment_id)

    # without out the fix, the following could would fail -> no assertions.
    run_experiment_from_schedule_and_malfunction(
        schedule_and_malfunction=schedule_and_malfunction,
        experiment_parameters=experiment_parameters,
        verbose=verbose,
        debug=debug,
    )
