import os
import time
from typing import Optional

from flatland.action_plan.action_plan_player import ControllerFromTrainrunsReplayerRenderCallback
from flatland.envs.rail_env import RailEnv


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def make_render_call_back_for_replay(env: RailEnv,
                                     rendering: bool = False) -> ControllerFromTrainrunsReplayerRenderCallback:
    if rendering:
        from flatland.utils.rendertools import AgentRenderVariant
        from flatland.utils.rendertools import RenderTool
        renderer = RenderTool(env, gl="PILSVG",
                              agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                              show_debug=True,
                              clear_debug_text=True,
                              screen_height=1000,
                              screen_width=1000)

    def render(*argv):
        if rendering:
            renderer.render_env(show=True, show_observations=False, show_predictions=False)
            time.sleep(2)

    return render


def init_renderer_for_env(env: RailEnv, rendering: bool = False):
    if rendering:
        from flatland.utils.rendertools import AgentRenderVariant
        from flatland.utils.rendertools import RenderTool
        return RenderTool(env, gl="PILSVG",
                          agent_render_variant=AgentRenderVariant.AGENT_SHOWS_OPTIONS_AND_BOX,
                          show_debug=True,
                          clear_debug_text=True,
                          screen_height=1000,
                          screen_width=1000)


def render_env(renderer, test_id: int, solver_name, i_step: int,
               image_output_directory: Optional[str] = './rendering_output'):
    if renderer is not None:
        from flatland.utils.rendertools import RenderTool
        renderer: RenderTool = renderer
        renderer.render_env(show=True, show_observations=False, show_predictions=False)
        if image_output_directory is not None:
            if not os.path.exists(image_output_directory):
                os.makedirs(image_output_directory)
            renderer.gl.save_image(os.path.join(image_output_directory,
                                                "flatland_frame_{:04d}_{:04d}_{}.png".format(test_id,
                                                                                             i_step,
                                                                                             solver_name)))


def cleanup_renderer_for_env(renderer):
    if renderer:
        from flatland.utils.rendertools import RenderTool
        renderer: RenderTool = renderer
        # close renderer window
        renderer.close_window()
