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


if __name__ == '__main__':
    base_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir)
    notebooks = [f for f in os.listdir(base_path) if f.endswith(".Rmd")]
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
            # tweak 4: do not use function that use plotlib
            if False:
                dest_text = re.sub('^(.*plot_route_dag)', r'#\g<1>', dest_text, flags=re.MULTILINE)
            dest_text = re.sub('^(.*disturbance_propagation_graph_visualization)', r'#\g<1>', dest_text, flags=re.MULTILINE)

            print(dest_text)
            multiline_eval(dest_text)
