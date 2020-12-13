import numpy as np

from rsp.global_data_configuration import EXPERIMENT_DATA_SUBDIRECTORY_NAME
from rsp.pipeline.rsp_pipeline import rsp_pipeline
from rsp.scheduling.schedule import save_schedule
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ExperimentParameters
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import InfrastructureParameters
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import InfrastructureParametersRange
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ReScheduleParameters
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ReScheduleParametersRange
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ScheduleParameters
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ScheduleParametersRange
from rsp.step_02_infrastructure_generation.infrastructure import create_env_from_experiment_parameters
from rsp.step_03_schedule_generation.schedule_generation import gen_schedule
from rsp.step_05_experiment_run.experiment_results import ExperimentResults
from rsp.step_05_experiment_run.experiment_results_analysis import ExperimentResultsAnalysis
from rsp.step_05_experiment_run.experiment_run import create_experiment_folder_name
from rsp.step_05_experiment_run.experiment_run import delete_experiment_folder
from rsp.step_05_experiment_run.experiment_run import gen_infrastructure
from rsp.step_05_experiment_run.experiment_run import load_and_expand_experiment_results_from_data_folder
from rsp.step_05_experiment_run.experiment_run import load_data_from_individual_csv_in_data_folder
from rsp.step_05_experiment_run.experiment_run import load_experiments_results
from rsp.step_05_experiment_run.experiment_run import load_infrastructure
from rsp.step_05_experiment_run.experiment_run import load_schedule
from rsp.step_05_experiment_run.experiment_run import run_experiment_in_memory


def test_rsp_pipeline():
    """Run a simple agenda and save and load the results.

    Check that loading gives the same result.
    """
    experiment_name = "test_rsp_pipeline"
    experiment_base_directory = "target/" + create_experiment_folder_name(experiment_name)
    try:
        experiment_folder_name, experiment_agenda = rsp_pipeline(
            experiment_base_directory=experiment_base_directory,
            experiment_name=experiment_name,
            infra_parameters_range=InfrastructureParametersRange(
                number_of_agents=[2, 2, 1],
                width=[30, 30, 1],
                height=[30, 30, 1],
                flatland_seed_value=[190, 190, 1],
                max_rail_in_city=[6, 6, 1],
                max_rail_between_cities=[2, 2, 1],
                max_num_cities=[20, 20, 1],
                number_of_shortest_paths_per_agent=[10, 10, 1],
            ),
            schedule_parameters_range=ScheduleParametersRange(asp_seed_value=[814, 814, 1], number_of_shortest_paths_per_agent_schedule=[1, 1, 1],),
            reschedule_parameters_range=ReScheduleParametersRange(
                earliest_malfunction=[20, 20, 1],
                malfunction_duration=[20, 20, 1],
                malfunction_agent_id=[0, 2, 2],
                number_of_shortest_paths_per_agent=[10, 10, 1],
                max_window_size_from_earliest=[60, 60, 1],
                asp_seed_value=[99, 99, 1],
                # route change is penalized the same as 30 seconds delay
                weight_route_change=[60, 60, 1],
                weight_lateness_seconds=[1, 1, 1],
                # Define the desired speed profiles
            ),
            # Define the desired speed profiles
            speed_data={1.0: 1},
            run_analysis=True,
        )

        # load results
        experiment_data_folder = f"{experiment_folder_name}/{EXPERIMENT_DATA_SUBDIRECTORY_NAME}"
        loaded_results, _ = load_and_expand_experiment_results_from_data_folder(experiment_data_folder_name=experiment_data_folder)

        # since we do not return the results in memory from run_experiment_agenda (SIM-393), do some sanity checks:
        assert len(loaded_results) == 2, len(loaded_results)
        loaded_result: ExperimentResultsAnalysis = loaded_results[0]
        assert loaded_result.results_online_unrestricted.solver_statistics is not None

        experiment_results = loaded_result
        experiment_parameters: ExperimentParameters = experiment_agenda.experiments[0]

        # check that asp seed value is received in solver
        assert experiment_results.results_schedule.solver_seed == experiment_parameters.schedule_parameters.asp_seed_value, (
            f"actual={experiment_results.results_schedule.solver_seed}, " f"expected={experiment_parameters.asp_seed_value}"
        )
        assert experiment_results.results_online_unrestricted.solver_seed == experiment_parameters.schedule_parameters.asp_seed_value, (
            f"actual={experiment_results.results_online_unrestricted.solver_seed}, " f"expected={experiment_parameters.asp_seed_value}"
        )
        assert experiment_results.results_offline_delta.solver_seed == experiment_parameters.schedule_parameters.asp_seed_value, (
            f"actual={experiment_results.results_offline_delta.solver_seed}, " f"expected={experiment_parameters.asp_seed_value}"
        )

        loaded_df = load_data_from_individual_csv_in_data_folder(experiment_data_folder_name=experiment_data_folder)
        assert len(loaded_df) == 2, len(loaded_df)

        assert load_experiments_results(experiment_data_folder_name=experiment_data_folder, experiment_id=0) is not None, None
    finally:
        delete_experiment_folder(experiment_base_directory)


def test_run_alpha_beta(regen_schedule: bool = False):
    """Ensure that we get the exact same solution if we multiply the weights
    for route change and lateness by the same factor."""

    experiment_parameters = ExperimentParameters(
        experiment_id=0,
        grid_id=0,
        infra_id_schedule_id=0,
        infra_parameters=InfrastructureParameters(
            infra_id=0,
            width=30,
            height=30,
            number_of_agents=11,
            flatland_seed_value=12,
            max_num_cities=20,
            grid_mode=True,
            max_rail_between_cities=2,
            max_rail_in_city=6,
            speed_data={1.0: 1.0, 0.5: 0.0, 0.3333333333333333: 0.0, 0.25: 0.0},
            number_of_shortest_paths_per_agent=10,
        ),
        schedule_parameters=ScheduleParameters(infra_id=0, schedule_id=0, asp_seed_value=94, number_of_shortest_paths_per_agent_schedule=1),
        re_schedule_parameters=ReScheduleParameters(
            earliest_malfunction=20,
            malfunction_duration=20,
            malfunction_agent_id=0,
            weight_route_change=20,
            weight_lateness_seconds=1,
            max_window_size_from_earliest=np.inf,
            number_of_shortest_paths_per_agent=10,
            asp_seed_value=94,
        ),
    )
    scale = 5
    experiment_parameters_scaled = ExperimentParameters(
        **dict(
            experiment_parameters._asdict(),
            **{
                "re_schedule_parameters": ReScheduleParameters(
                    **dict(
                        experiment_parameters.re_schedule_parameters._asdict(),
                        **{
                            "weight_route_change": experiment_parameters.re_schedule_parameters.weight_route_change * scale,
                            "weight_lateness_seconds": experiment_parameters.re_schedule_parameters.weight_lateness_seconds * scale,
                        },
                    )
                )
            },
        )
    )

    static_rail_env = create_env_from_experiment_parameters(experiment_parameters.infra_parameters)
    static_rail_env.load_resource("tests.02_regression_tests.data.alpha_beta", "static_env_alpha_beta.pkl")

    # since schedule generation is not deterministic, we need to pickle the output of A.2 experiment setup
    # regen_schedule to fix the regression test in case of breaking API change in the pickled content
    if regen_schedule:
        infra_scaled = gen_infrastructure(infra_parameters=experiment_parameters_scaled)
        schedule_scaled = gen_schedule(infrastructure=infra_scaled, schedule_parameters=experiment_parameters_scaled.schedule_parameters)

        infra = gen_infrastructure(infra_parameters=experiment_parameters)
        schedule = gen_schedule(infrastructure=infra, schedule_parameters=experiment_parameters.schedule_parameters)
        save_schedule(
            schedule=schedule_scaled,
            schedule_parameters=experiment_parameters_scaled.schedule_parameters,
            base_directory="tests/02_regression_tests/data/alpha_beta",
        )
        save_schedule(
            schedule=schedule, schedule_parameters=experiment_parameters.schedule_parameters, base_directory="tests/02_regression_tests/data/alpha_beta"
        )
    infra_scaled, _ = load_infrastructure(base_directory="tests/02_regression_tests/data/alpha_beta", infra_id=0)
    infra, _ = load_infrastructure(base_directory="tests/02_regression_tests/data/alpha_beta", infra_id=0)
    schedule_scaled, _ = load_schedule(base_directory="tests/02_regression_tests/data/alpha_beta", infra_id=0)
    schedule, _ = load_schedule(base_directory="tests/02_regression_tests/data/alpha_beta", infra_id=0, schedule_id=0)

    experiment_result_scaled: ExperimentResults = run_experiment_in_memory(
        schedule=schedule_scaled, experiment_parameters=experiment_parameters_scaled, infrastructure_topo_dict=infra_scaled.topo_dict
    )

    experiment_result: ExperimentResults = run_experiment_in_memory(
        schedule=schedule, experiment_parameters=experiment_parameters, infrastructure_topo_dict=infra.topo_dict
    )

    # although re-scheduling is not deterministic, it should produce solutions with the same costs
    costs_online_unrestricted = experiment_result.results_online_unrestricted.optimization_costs
    assert costs_online_unrestricted > 0
    costs_online_unrestricted_scaled = experiment_result_scaled.results_online_unrestricted.optimization_costs
    assert costs_online_unrestricted * scale == costs_online_unrestricted_scaled
    assert experiment_result.results_online_unrestricted.trainruns_dict == experiment_result_scaled.results_online_unrestricted.trainruns_dict
