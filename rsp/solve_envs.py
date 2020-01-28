"""Run scheduling problem from files."""
import os

from rsp.solve_utils import HEADER
from rsp.solve_utils import list_files
from rsp.solve_utils import load_flatland_environment_from_file_with_fixed_seed
from rsp.solve_utils import test_helper


# ----------------------------- Helper -----------------------------------------------------------


def main():
    rendering = False

    def load_environment(loop_index, loop_item):
        env = load_flatland_environment_from_file_with_fixed_seed(loop_item)
        return env, env.width, env.height, env.get_num_agents()

    all_files = list_files(os.path.join(os.path.dirname(__file__), './../Envs/Round_2/'))

    sbb_results = open("solve_envs.csv", "w")
    sbb_results.writelines(HEADER)

    output_file_name = "solve_envs.csv"
    test_helper(load_environment, output_file_name, rendering, all_files)


if __name__ == '__main__':
    main()
