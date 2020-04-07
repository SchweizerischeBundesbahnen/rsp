import ast
import os
import re
import sys

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


def test_notebook_runs_trourgh():
    # skip under windows because of windows being opened
    if sys.platform == "win32":
        return
    base_path = os.path.join(__file__, "..", "..", "..")
    notebooks = [f for f in os.listdir(base_path) if f.endswith(".Rmd")]
    for notebook in notebooks:
        print("*****************************************************************")
        print("Converting and running {}".format(notebook))
        print("*****************************************************************")

        with open(os.path.join(base_path, notebook)) as file_in:
            notebook = read(file_in)

            dest_text = writes(notebook, fmt="py:percent")

            # tweak 1: skip display
            dest_text = re.sub('^display', "#display", dest_text, flags=re.MULTILINE)
            # tweak 2: skip plot_route_dag (window has to be closed manually)
            dest_text = re.sub('^plot_route_dag', "#plot_route_dag", dest_text, flags=re.MULTILINE)
            print(dest_text)
            multiline_eval(dest_text)
