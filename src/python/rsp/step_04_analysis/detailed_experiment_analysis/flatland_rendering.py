import os
from pathlib import Path

from rsp.step_03_run.experiment_results_analysis import ExperimentResultsAnalysis
from rsp.step_03_run.experiments import create_env_from_experiment_parameters
from rsp.step_03_run.experiments import EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME
from rsp.utils.file_utils import check_create_folder
from rsp.utils.flatland_replay_utils import render_trainruns


def render_flatland_env(  # noqa
    data_folder: str, experiment_data: ExperimentResultsAnalysis, experiment_id: int, render_schedule: bool = True, render_reschedule: bool = True
):
    """
    Method to render the environment for visual inspection
    Parameters
    ----------
render_flatland_env
    data_folder: str
        Folder name to store and load images from
    experiment_data: ExperimentResultsAnalysis
        experiment data used for visualization
    experiment_id: int
        ID of experiment we like to visualize
    render_reschedule
    render_schedule

    Returns
    -------
    File paths to generated videos to render in the notebook
    """

    # Generate environment for rendering
    rail_env = create_env_from_experiment_parameters(experiment_data.experiment_parameters.infra_parameters)

    # Generate aggregated visualization
    output_folder = f"{data_folder}/{EXPERIMENT_ANALYSIS_SUBDIRECTORY_NAME}/"
    check_create_folder(output_folder)
    video_src_schedule = None
    video_src_reschedule = None

    # Generate the Schedule video
    if render_schedule:
        # Import the generated video
        title = "Schedule"
        video_src_schedule = os.path.join(
            output_folder,
            f"experiment_{experiment_data.experiment_id:04d}_analysis",
            f"experiment_{experiment_data.experiment_id}_rendering_output_{title}/",
            f"experiment_{experiment_id}_flatland_data_analysis.mp4",
        )

        # Only render if file is not yet created
        if not os.path.exists(video_src_schedule):
            render_trainruns(
                data_folder=output_folder,
                experiment_id=experiment_data.experiment_id,
                title=title,
                rail_env=rail_env,
                trainruns=experiment_data.solution_full,
                convert_to_mpeg=True,
            )

    # Generate the Reschedule video
    if render_reschedule:
        # Import the generated video
        title = "Reschedule"
        video_src_reschedule = os.path.join(
            output_folder,
            f"experiment_{experiment_data.experiment_id:04d}_analysis",
            f"experiment_{experiment_data.experiment_id}_rendering_output_{title}/",
            f"experiment_{experiment_id}_flatland_data_analysis.mp4",
        )
        # Only render if file is not yet created
        if not os.path.exists(video_src_reschedule):
            render_trainruns(
                data_folder=output_folder,
                experiment_id=experiment_data.experiment_id,
                malfunction=experiment_data.malfunction,
                title=title,
                rail_env=rail_env,
                trainruns=experiment_data.solution_full_after_malfunction,
                convert_to_mpeg=True,
            )

    return Path(video_src_schedule) if render_schedule else None, Path(video_src_reschedule) if render_reschedule else None
