import ast
import os
import re
from unittest import skip

from jupytext import read
from jupytext import writes


# https://stackoverflow.com/questions/12698028/why-is-pythons-eval-rejecting-this-multiline-string-and-how-can-i-fix-it
def multiline_eval(expr):
    """Evaluate several lines of input, returning the result of the last
    line."""
    tree = ast.parse(expr)
    eval_expr = ast.Expression(tree.body[-1].value)
    exec_expr = ast.Module(tree.body[:-1])
    exec(compile(exec_expr, "file", "exec"))
    return eval(compile(eval_expr, "file", "eval"))


@skip
def test_notebooks_run_through():
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir, "src", "jupyter")
    # TODO SIM-571 SIM-672 activate potassco notebooks when data is added to src.python.rsp-data
    notebooks = [f for f in os.listdir(base_path) if f.endswith(".Rmd")]
    for notebook in notebooks:
        print("*****************************************************************")
        print("Converting and running {}".format(notebook))
        print("*****************************************************************")

        with open(os.path.join(base_path, notebook)) as file_in:
            notebook = read(file_in)

            dest_text = writes(notebook, fmt="py:percent")

            # tweak 1: print instead of display
            dest_text = re.sub("^display", "print", dest_text, flags=re.MULTILINE)
            # tweak 2: use plot_route_dag with save=True (in order to prevent plt from opening window in ci)
            dest_text = re.sub("^(plot_route_dag.*)\\)", r"\g<1>, save=True)", dest_text, flags=re.MULTILINE)
            # tweak 3: do not show Video
            dest_text = re.sub("^Video", r"#Video", dest_text, flags=re.MULTILINE)

            # tweak 4: run only a subset of all data in compute_time_analysis.Rmd in integration tests
            dest_text = re.sub("^experiments_of_interest =.*", """experiments_of_interest = range(10)""", dest_text, flags=re.MULTILINE)

            print(dest_text)
            multiline_eval(dest_text)
