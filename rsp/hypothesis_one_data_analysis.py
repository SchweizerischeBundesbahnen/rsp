"""Analysis of the experiment data for hypothesis one.

Hypothesis 1:
    We can compute good recourse actions, i.e., an adapted plan within the time budget,
    if all the variables are fixed, except those related to services that are affected by the
    disruptions implicitly or explicitly.

Hypothesis 2:
    Machine learning can predict services that are affected by disruptions implicitly or
    explicitly. Hypothesis 3: If hypothesis 2 is true, in addition, machine
    learning can predict the state of the system in the next time period
    after re-scheduling.
"""
from flatland.action_plan.action_plan import ControllerFromTrainruns
from flatland.envs.rail_env_shortest_paths import get_k_shortest_paths
from flatland.envs.rail_trainrun_data_structures import TrainrunDict
from pandas import DataFrame

from rsp.utils.analysis_tools import average_over_trials
from rsp.utils.analysis_tools import three_dimensional_scatter_plot
from rsp.utils.data_types import ExperimentAgenda
from rsp.utils.data_types import ExperimentFreezeDict
from rsp.utils.data_types import ExperimentParameters
from rsp.utils.data_types import visualize_experiment_freeze
from rsp.utils.experiment_utils import replay
from rsp.utils.experiments import create_env_pair_for_experiment
from rsp.utils.experiments import load_experiment_agenda_from_file
from rsp.utils.experiments import load_experiment_results_from_folder


def render_experiment(experiment: ExperimentParameters, data_frame: DataFrame):
    """Render the experiment in the analysis.

    Parameters
    ----------
    experiment: ExperimentParameters
        experiment parameters for all trials
    data_frame: DataFrame
        Pandas data frame with one ore more trials of this experiment.
    """
    from rsp.utils.experiment_solver import RendererForEnvInit, RendererForEnvRender, RendererForEnvCleanup

    # TODO SIM-251 display experiment_id in FLATland replay?
    print(experiment.experiment_id)

    # find first row for this experiment (iloc[0]
    rows = data_frame.loc[data_frame['experiment_id'] == experiment.experiment_id]

    static_rail_env, malfunction_rail_env = create_env_pair_for_experiment(experiment)
    train_runs: TrainrunDict = rows['solution_full_after_malfunction'].iloc[0]

    # TODO SIM-239 add all experiment_freezes to experiment results and rename
    experiment_freeze_delta: ExperimentFreezeDict = rows['experiment_freeze'].iloc[0]

    # TODO SIM-239 add k to experiment params
    k = 10
    # TODO SIM-239 add agents paths dict to experiment results
    agents_paths_dict = {
        i: get_k_shortest_paths(static_rail_env,
                                agent.initial_position,
                                agent.initial_direction,
                                agent.target,
                                k) for i, agent in enumerate(static_rail_env.agents)
    }

    for agent_id in experiment_freeze_delta:
        visualize_experiment_freeze(
            agent_paths=agents_paths_dict[agent_id],
            f=experiment_freeze_delta[agent_id],
            title=f"experiment {experiment.experiment_id} of agent {agent_id}"
        )

    controller_from_train_runs = ControllerFromTrainruns(malfunction_rail_env, train_runs)

    # TODO SIM-239 should come from parameter
    rendering = True
    if rendering:
        from rsp.utils.experiment_render_utils import cleanup_renderer_for_env
        from rsp.utils.experiment_render_utils import render_env
        from rsp.utils.experiment_render_utils import init_renderer_for_env

        init_renderer_for_env = init_renderer_for_env
        render_renderer_for_env = render_env
        cleanup_renderer_for_env = cleanup_renderer_for_env
    else:
        init_renderer_for_env: RendererForEnvInit = lambda *args, **kwargs: None
        render_renderer_for_env: RendererForEnvRender = lambda *args, **kwargs: None
        cleanup_renderer_for_env: RendererForEnvCleanup = lambda *args, **kwargs: None

    renderer = init_renderer_for_env(malfunction_rail_env, rendering)

    def rendering_call_back(test_id: int, solver_name, i_step: int):
        render_renderer_for_env(renderer, test_id, solver_name, i_step)

    replay(
        controller_from_train_runs=controller_from_train_runs,
        env=malfunction_rail_env,
        stop_on_malfunction=False,
        solver_name='data_analysis',
        disable_verification_in_replay=False,
        rendering_call_back=rendering_call_back
    )
    cleanup_renderer_for_env(renderer)


if __name__ == '__main__':
    # Import the desired experiment results
    data_folder = './exp_hypothesis_one_2020_01_24T09_19_03'
    experiment_data: DataFrame = load_experiment_results_from_folder(data_folder)
    experiment_agenda: ExperimentAgenda = load_experiment_agenda_from_file(data_folder)

    for key in ['size', 'n_agents', 'max_num_cities', 'max_rail_between_cities', 'max_rail_in_city']:
        experiment_data[key] = experiment_data[key].astype(float)

    # Average over the trials of each experiment
    averaged_data, std_data = average_over_trials(experiment_data)

    # # Initially plot the computation time vs the level size and the number of agent
    three_dimensional_scatter_plot(data=averaged_data, error=std_data, columns=['n_agents', 'size', 'time_full'])

    three_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                   columns=['n_agents', 'size', 'time_full_after_malfunction'])
    three_dimensional_scatter_plot(data=averaged_data, error=std_data,
                                   columns=['n_agents', 'size', 'time_delta_after_malfunction'])

    # TODO SIM-251 filter which to be replayed
    for experiment in experiment_agenda.experiments:
        render_experiment(experiment=experiment, data_frame=experiment_data)
