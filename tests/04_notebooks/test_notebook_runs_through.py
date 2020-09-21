import ast
import os
import re

from jupytext import read
from jupytext import writes


# https://stackoverflow.com/questions/12698028/why-is-pythons-eval-rejecting-this-multiline-string-and-how-can-i-fix-it
def multiline_eval(expr):
    """Evaluate several lines of input, returning the result of the last
    line."""
    tree = ast.parse(expr)
    eval_expr = ast.Expression(tree.body[-1].value)
    exec_expr = ast.Module(tree.body[:-1])
    exec(compile(exec_expr, 'file', 'exec'))
    return eval(compile(eval_expr, 'file', 'eval'))


def test_notebooks_run_through():
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
    # TODO SIM-571 activate potassco notebooks when data is added to rsp-data
    notebooks = [f for f in os.listdir(base_path) if f.endswith(".Rmd") and 'potassco' not in f]
    for notebook in notebooks:
        print("*****************************************************************")
        print("Converting and running {}".format(notebook))
        print("*****************************************************************")

        with open(os.path.join(base_path, notebook)) as file_in:
            notebook = read(file_in)

            dest_text = writes(notebook, fmt="py:percent")

            # tweak 1: print instead of display
            dest_text = re.sub('^display', "print", dest_text, flags=re.MULTILINE)
            # tweak 2: use plot_route_dag with save=True (in order to prevent plt from opening window in ci)
            dest_text = re.sub('^(plot_route_dag.*)\\)', r'\g<1>, save=True)', dest_text, flags=re.MULTILINE)
            # tweak 3: do not show Video
            dest_text = re.sub('^Video', r'#Video', dest_text, flags=re.MULTILINE)

            # tweak 4: insert
            # TODO SIM-672 global configuration?
            dest_text = re.sub('^experiment_base_directory =.*',
                               """experiment_base_directory = '../rsp-data/h1_2020_08_24T21_04_42_dummydata_2020_09_11T10_28_31'""", dest_text,
                               flags=re.MULTILINE)
            dest_text = re.sub('^experiment_of_interest =.*',
                               """experiment_of_interest = 0""", dest_text,
                               flags=re.MULTILINE)

            print(dest_text)
            multiline_eval(dest_text)
