from rsp.experiment_solvers.experiment_solver import ASPExperimentSolver
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import ExperimentResults
from rsp.utils.experiments import create_env_pair_for_experiment
from rsp.utils.experiments import load_experiment_agenda_from_file
from rsp.utils.experiments import load_schedule_and_malfunction


def test_exp_006_hypothesis_window_size_null_2020_03_17T18_34_47_experiment_158(
        verbose=True,
        rendering=True,
        debug=True):
    experiment_agenda_directory = "data/exp_006_hypothesis_window_size_null_2020_03_17T18_34_47/agenda"
    experiment_agenda = load_experiment_agenda_from_file(experiment_agenda_directory)
    schedule_and_malfunction = load_schedule_and_malfunction(
        experiment_agenda_directory=experiment_agenda_directory,
        experiment_id=158)
    experiment_parameters: ExperimentParameters = experiment_agenda.experiments[158]
    _, malfunction_rail_env = create_env_pair_for_experiment(experiment_parameters)
    from flatland.utils.rendertools import RenderTool, AgentRenderVariant
    env_renderer = RenderTool(malfunction_rail_env, gl="PILSVG",
                              agent_render_variant=AgentRenderVariant.ONE_STEP_BEHIND,
                              show_debug=False,
                              screen_height=600,  # Adjust these parameters to fit your resolution
                              screen_width=800)
    env_renderer.reset()
    env_renderer.render_env(show=True, show_observations=False, show_predictions=False)

    # wrap reset params in this function, so we avoid copy-paste errors each time we have to reset the malfunction_rail_env
    def malfunction_env_reset():
        malfunction_rail_env.reset(False, False, False, experiment_parameters.flatland_seed_value)

    malfunction_env_reset()

    # B2: full and delta re-scheduling
    solver = ASPExperimentSolver()
    experiment_results: ExperimentResults = solver._run_experiment_from_environment(
        schedule_and_malfunction=schedule_and_malfunction,
        malfunction_rail_env=malfunction_rail_env,
        malfunction_env_reset=malfunction_env_reset,
        experiment_parameters=experiment_parameters,
        verbose=verbose,
        debug=debug
    )
    if experiment_results is None:
        print(f"No malfunction for experiment {experiment_parameters.experiment_id}")
        return []

    if rendering:
        from flatland.utils.rendertools import RenderTool, AgentRenderVariant
        env_renderer.close_window()
