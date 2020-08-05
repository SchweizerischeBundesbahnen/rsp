import json
import time
from threading import Timer
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Set

import clingo
import numpy as np
from importlib_resources import path

from rsp.experiment_solvers.asp import theory
from rsp.experiment_solvers.asp.data_types import ASPHeuristics
from rsp.experiment_solvers.asp.data_types import ASPObjective
from rsp.logger import rsp_logger
from rsp.logger import VERBOSE
from rsp.utils.global_constants import DL_PROPAGATE_PARTIAL

FluxHelperResult = NamedTuple('FluxHelperResult', [
    # TODO SIM-121 asp_solver should use proper data structures instead of strings to represent answer sets
    ('answer_sets', List[Set[str]]),
    ('stats', Dict),
    # future use for incremental solving?
    ('ctl', clingo.Control),
    ('dl', theory.Theory),
    ('asp_seed_value', Optional[int])
])


def flux_helper(
        asp_data: List[str],
        asp_objective: ASPObjective = ASPObjective.MINIMIZE_SUM_RUNNING_TIMES,
        asp_heuristics: Optional[List[ASPHeuristics]] = None,
        asp_seed_value: int = 94,
        nb_threads: int = 2,
        no_optimize: bool = False,
        verbose: bool = False,
        debug: bool = False
) -> FluxHelperResult:
    """Includes the necessary encodings and calls `_asp_helper` with them.

    Parameters
    ----------


    asp_data
        data part
    asp_objective
        which asp objective should be applied if any

    nb_threads
    asp_seed_value
    asp_heuristics
        which heuristics to apply?
    no_optimize
        do not optimize
    debug
    verbose

    Returns
    -------
    """
    prg_text_joined = "\n".join(asp_data)
    if debug:
        print(prg_text_joined)

    with path('res.asp.encodings', 'encoding.lp') as encoding_path:
        paths = [encoding_path]
    rsp_logger.info(f"asp_heuristics={asp_heuristics}")
    if asp_heuristics:
        for asp_heurisic in asp_heuristics:
            with path('res.asp.encodings', f'{asp_heurisic.value}.lp') as heuristic_routes_path:
                paths.append(heuristic_routes_path)
    if asp_objective and not no_optimize:
        with path('res.asp.encodings', f'{asp_objective.value}.lp') as objetive_path:
            paths.append(objetive_path)
        if asp_objective in [ASPObjective.MINIMIZE_DELAY, ASPObjective.MINIMIZE_DELAY_ROUTES_COMBINED]:
            with path('res.asp.encodings', f'delay_linear_within_one_minute.lp') as delay_model_path:
                paths.append(delay_model_path)

    flux_result = _asp_helper(
        encoding_files=paths,
        plain_encoding=prg_text_joined,
        asp_seed_value=asp_seed_value,
        nb_threads=nb_threads,
        verbose=verbose,
        debug=debug,
        no_optimize=no_optimize,
        asp_heuristics=asp_heuristics
    )

    return flux_result


# snippets from https://code.sbb.ch/projects/TP_TMS_PAS/repos/kapaplan-asp/browse/src/solver/clingo_controller.py
def _asp_helper(encoding_files: List[str],  # noqa: C901
                plain_encoding: Optional[str] = None,
                verbose: bool = False,
                debug: bool = False,
                nb_threads: int = 2,
                no_optimize: bool = False,
                asp_heuristics: List[ASPHeuristics] = None,
                asp_seed_value: Optional[int] = None) -> FluxHelperResult:
    """Runs clingo-dl with in the desired mode.
    Parameters
    ----------
    encoding_files
        encodings as file list to load
    plain_encoding
        plain encoding as string
    verbose
        prints a lot to debug
    """
    # Info Max Ostrovski 2019-11-20: die import dl Variante
    # (https://www.cs.uni-potsdam.de/~torsten/hybris.pdf  Listing 1.8 line 9)
    # bezieht sich auf eine sehr alte clingo[DL] version. Im Rahmen einer einheitlichen API für alle clingo Erweiterungen
    # (clingo[DL], clingcon, clingo[LP]) ist die neue Variante mit der python theory zu verwenden.
    rsp_logger.info(f"no_optimize={no_optimize}")

    dl = theory.Theory("clingodl", "clingo-dl")
    if DL_PROPAGATE_PARTIAL:
        rsp_logger.info("running with --propagate=partial")
        dl.configure_propagator("propagate", "partial")
    else:
        rsp_logger.info("running with --propagate=partial")
    ctl_args = [f"-t{nb_threads}", "--lookahead=no"]

    if asp_seed_value is not None:
        ctl_args.append(f"--seed={asp_seed_value}")
    if asp_heuristics is not None:
        ctl_args.append(f"--heur=domain")

    ctl = clingo.Control(ctl_args)
    rsp_logger.info(f"{ctl_args}")

    # find optimal model; if not optimizing, find all models!
    ctl.configuration.solve.models = 0
    # find only first optimal model
    ctl.configuration.solve.opt_mode = 'opt'
    dl.register_propagator(ctl)

    if verbose:
        rsp_logger.log("taking encodings from {}".format(encoding_files), level=VERBOSE)
        if plain_encoding and debug:
            print("taking plain_encoding={}".format(plain_encoding))
    for enc in encoding_files:
        ctl.load(str(enc))
    if plain_encoding:
        ctl.add("base", [], plain_encoding)
    rsp_logger.log(VERBOSE, "Grounding starting...")
    rsp_logger.info("Grounding starting...")
    grounding_start_time = time.time()
    ctl.ground([("base", [])])

    rsp_logger.info("Grounding took {}s".format(time.time() - grounding_start_time))

    all_answers = _asp_loop(ctl=ctl, dl=dl, no_optimize=no_optimize, verbose=verbose, debug=debug)
    statistics: Dict = ctl.statistics

    if verbose:
        print(all_answers)
        _print_configuration(ctl)
        _print_stats(statistics)

    # SIM-429 assert that our models are tight (sccs==0)
    assert statistics["problem"]["lp"][
               "sccs"] == 0, f'not tight statistics["problem"]["lp"]["sccs"]={statistics["problem"]["lp"]["sccs"]}'

    return FluxHelperResult(all_answers, statistics, ctl, dl, asp_seed_value)


def _asp_loop(ctl: clingo.Control,  # noqa: C901
              dl: theory.Theory,
              no_optimize: bool = False,
              verbose: bool = False,
              debug: bool = False,
              timeout: int = 10 * 60 * 60):
    """Loop over models coming from the ASP solve call until optimal one found
    and return the first optimal.

    Parameters
    ----------
    ctl
    dl
    no_optimize
    verbose
    debug
    timeout
        interrupt the solver after this time. Solving only, does not cover grounding.

    Returns
    -------
    """
    all_answers = []
    min_cost = np.inf
    timer = None

    def on_model(model):
        try:
            nonlocal all_answers, min_cost, timer, no_optimize, dl
            if len(model.cost) > 0:
                cost = model.cost[0]
                if cost < min_cost:
                    rsp_logger.log(VERBOSE, "Optimization: {}".format(cost))
                    min_cost = cost
                    all_answers = []
            # TODO SIM-121 convert to handable data structures instead of strings!
            sol = str(model).split(" ")
            if debug:
                for v in sol:
                    print(v)
            for name, value in dl.assignment(model.thread_id):
                v = f"dl({name},{value})"
                sol.append(v)
                if debug:
                    print(v)
            all_answers.append(frozenset(sol))
            timer.cancel()
            timer = Timer(interval=timer.interval, function=timer.function)
            timer.start()
            if no_optimize:
                return False
        except Exception as e:
            print(str(e))

    interrupted = False

    def interrupt():
        nonlocal ctl, interrupted
        interrupted = True
        ctl.interrupt()

    with ctl.solve(async_=True, on_statistics=dl.on_statistics, on_model=on_model) as handle:
        # the timeout given to handle.wait() seems not to work with clingo-dl -> use timer instead
        # TODO check with Potsdam why handle.wait() does not work as expected: https://potassco.org/clingo/python-api/5.4/
        timer = Timer(interval=timeout, function=interrupt)
        timer.start()
        while not handle.wait():
            pass
    if len(all_answers) == 0:
        _print_stats(statistics=ctl.statistics)
        raise ValueError(f"ASP solver: No solution found. Interrupted={interrupted}")
    return all_answers


def _print_stats(statistics, print_full_statistics: bool = False):
    if print_full_statistics:
        print("=================================================================================")
        print("= FULL STATISTICS                                                               =")
        print("=================================================================================")
        print(json.dumps(statistics, sort_keys=True, indent=4, separators=(',', ': ')))
        print("")
    print("=================================================================================")
    print("= SUMMARY                                                                       =")
    print("=================================================================================")
    print("Models       : {:.0f}".format(statistics["summary"]["models"]["enumerated"]))
    print("Optimum      : {:3}".format("yes" if statistics["summary"]["models"]["optimal"] else "no"))
    print("Optimization : {}".format(statistics["summary"]["costs"]))
    print("Calls        : {}".format(statistics["summary"]["call"]))
    print("Time         : {:5.3f}s (Solving: {}s 1st Model: {}s Unsat: {}s Grounding etc. {}s)"
          .format(statistics["summary"]["times"]["total"],
                  statistics["summary"]["times"]["solve"],
                  statistics["summary"]["times"]["sat"],
                  statistics["summary"]["times"]["unsat"],
                  statistics["summary"]["times"]["total"] - statistics["summary"]["times"]["solve"]
                  ))
    percentage_solving_time = \
        100 * statistics["summary"]["times"]["solve"] / (statistics["summary"]["times"]["total"]
                                                         if statistics["summary"]["times"]["total"] != 0 else 1)
    print("Solving time : {:5.1f}%".format(percentage_solving_time))
    print("CPU Time     : {:5.3f}s".format(statistics["summary"]["times"]["cpu"]))
    print("=================================================================================")
    print("")
    print("=================================================================================")
    print("= SOLVER STATISTICS                                                           =")
    print("=================================================================================")
    print("Models       : {:.0f}".format(statistics["summary"]["models"]["enumerated"]))
    print("Choices      : {:.0f}".format(statistics["solving"]["solvers"]["choices"]))
    print("Conflicts    : {:.0f}".format(statistics["solving"]["solvers"]["conflicts"]))
    print("Backjumps    : {:.0f}".format(statistics["solving"]["solvers"]["conflicts_analyzed"]))
    print("Restarts     : {:.0f}".format(statistics["solving"]["solvers"]["restarts"]))
    print("")
    print("=================================================================================")
    print("= GROUNDNG OUTPUT STATISTICS                                                    =")
    print("=================================================================================")
    print("Rules        : {:.0f}".format(statistics["problem"]["lp"]["rules"]))
    print("  Choice     : {:.0f}".format(statistics["problem"]["lp"]["rules_choice"]))
    print("  Minimize   : {:.0f}".format(statistics["problem"]["lp"]["rules_minimize"]))
    print("Atoms        : {:.0f}".format(statistics["problem"]["lp"]["atoms"]))
    print("Tight        : {:3}".format("yes" if statistics["problem"]["lp"]["sccs"] == 0 else "no"))
    print("Variables    : {:.0f}".format(statistics["problem"]["generator"]["vars"]))
    print("Constraints  : {:.0f}".format(statistics["problem"]["generator"]["constraints"]))
    print("  Binary     : {:.0f}".format(statistics["problem"]["generator"]["constraints_binary"]))
    print("  Ternary    : {:.0f}".format(statistics["problem"]["generator"]["constraints_ternary"]))


def _print_configuration(ctl):
    print("")
    print("=================================================================================")
    print("= CONFIGRUATION                                                                 =")
    print("=================================================================================")
    for _, k in enumerate(ctl.configuration.solve.keys):
        print("{}={}\n  {}: {}"
              .format(k,
                      getattr(ctl.configuration.solve, k), k,
                      getattr(ctl.configuration.solve, "__desc_" + k))
              )
    print("")


def configuration_as_dict_from_control(ctl):
    configuration_as_dict = {k: str(getattr(ctl.configuration.solve, k)) for _, k in
                             enumerate(ctl.configuration.solve.keys)}
    return configuration_as_dict
