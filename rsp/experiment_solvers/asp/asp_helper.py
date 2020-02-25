import json
import time
from enum import Enum
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Set

import clingo
import numpy as np
from importlib_resources import path

from rsp.experiment_solvers.asp import theory


class ASPObjective(Enum):
    """enum value (key arbitrary) must be the same as encoding to be
    included."""

    # minimize_total_sum_of_running_times.lp
    MINIMIZE_SUM_RUNNING_TIMES = "minimize_total_sum_of_running_times"

    # minimize delay with respect to earliest constraints
    MINIMIZE_DELAY = "minimize_delay"

    # minimize route section penalties
    MINIMIZE_ROUTES = "minimize_routes"

    # minimize linear combination of route section penalties and delay
    MINIMIZE_DELAY_ROUTES_COMBINED = "minimize_delay_and_routes_combined"


class ASPHeuristics(Enum):
    """enum value (key arbitrary) must be the same as encoding to be
    included."""

    # avoiding delay at earlier nodes in the paths.
    # NOT USED YET (we do not give the data in re-scheduling yet)
    HEURISTIC_DELAY = "heuristic_DELAY"

    # tries to avoid routes where there is a penalty.
    # NOT USED YET (we do not give the data in re-scheduling yet)
    HEURISIC_ROUTES = "heuristic_ROUTES"

    # attempts to order conflicting trains by their possible arrival times at the edges where the conflict is located.
    # NOT USED YET (we do not give the data in re-scheduling yet)
    HEURISTIC_SEQ = "heuristic_SEQ"


FluxHelperResult = NamedTuple('FluxHelperResult', [
    # TODO SIM-121 asp_solver should use proper data structures instead of strings to represent answer sets
    ('answer_sets', List[Set[str]]),
    ('stats', Dict),
    # future use for incremental solving?
    ('ctl', clingo.Control),
    ('dl', theory.Theory),
])


def flux_helper(
        asp_data: List[str],
        bound_all_events: Optional[int] = None,
        asp_objective: ASPObjective = ASPObjective.MINIMIZE_SUM_RUNNING_TIMES,
        asp_heurisics: List[ASPHeuristics] = None,
        verbose: bool = False,
        debug: bool = False
) -> FluxHelperResult:
    """Includes the necessary encodings and calls `_asp_helper` with them.

    Parameters
    ----------
    asp_data
        data part
    bound_all_events
        upper bound on all arrival times
    asp_heuristics
        which heuristics to apply?
    asp_objective
        which asp objective should be applied if any

    Returns
    -------
    """
    prg_text_joined = "\n".join(asp_data)
    if debug:
        print(prg_text_joined)

    with path('res.asp.encodings', 'encoding.lp') as encoding_path:
        paths = [encoding_path]

    if asp_heurisics:
        for asp_heurisic in asp_heurisics:
            # TODO SIM-176 switch on heuristics
            if asp_heurisic in [ASPHeuristics.HEURISTIC_SEQ,
                                ASPHeuristics.HEURISTIC_DELAY,
                                ASPHeuristics.HEURISIC_ROUTES]:
                continue
            with path('res.asp.encodings', f'{asp_heurisic.value}.lp') as heuristic_routes_path:
                paths.append(heuristic_routes_path)
    if asp_objective:
        with path('res.asp.encodings', f'{asp_objective.value}.lp') as objetive_path:
            paths.append(objetive_path)
        if asp_objective in [ASPObjective.MINIMIZE_DELAY, ASPObjective.MINIMIZE_DELAY_ROUTES_COMBINED]:
            with path('res.asp.encodings', f'delay_linear_within_one_minute.lp') as delay_model_path:
                paths.append(delay_model_path)

    flux_result = _asp_helper(
        encoding_files=paths,
        bound_all_events=bound_all_events,
        plain_encoding=prg_text_joined,
        verbose=verbose,
        debug=debug
    )

    return flux_result


# snippets from https://code.sbb.ch/projects/TP_TMS_PAS/repos/kapaplan-asp/browse/src/solver/clingo_controller.py
def _asp_helper(encoding_files: List[str],
                plain_encoding: Optional[str] = None,
                verbose: bool = False,
                debug: bool = False,
                bound_all_events: Optional[int] = None,
                deterministic_mode: bool = True) -> FluxHelperResult:
    """Runs clingo-dl with in the desired mode.

    Parameters
    ----------
    encoding_files
        encodings as file list to load
    plain_encoding
        plain encoding as string
    verbose
        prints a lot to debug
    bound_all_events
        should the times have a global upper bound?
    asp_objective
        does multi or one-shot optimization depending on the objective
    deterministic_mode
        in deterministic mode, a seed is injected and multi-threading is deactivated
    """

    # Info Max Ostrovski 2019-11-20: die import dl Variante
    # (https://www.cs.uni-potsdam.de/~torsten/hybris.pdf  Listing 1.8 line 9)
    # bezieht sich auf eine sehr alte clingo[DL] version. Im Rahmen einer einheitlichen API für alle clingo Erweiterungen
    # (clingo[DL], clingcon, clingo[LP]) ist die neue Variante mit der python theory zu verwenden.
    dl = theory.Theory("clingodl", "clingo-dl")
    dl.configure_propagator("propagate", "partial")
    ctl_args = [f"-t1", "--lookahead=no"]

    if deterministic_mode:
        ctl_args = ["--seed=94", "-c use_decided=1", "-t1", "--lookahead=no"]
    ctl = clingo.Control(ctl_args)

    # find optimal model; if not optimizing, find all models!
    ctl.configuration.solve.models = 0
    # find only first optimal model
    ctl.configuration.solve.opt_mode = 'opt'
    dl.register_propagator(ctl)

    if verbose:
        print("taking encodings from {}".format(encoding_files))
        if plain_encoding:
            print("taking plain_encoding={}".format(plain_encoding))
    for enc in encoding_files:
        ctl.load(str(enc))
    if plain_encoding:
        ctl.add("base", [], plain_encoding)
    if verbose:
        print("Grounding starting...")
    grounding_start_time = time.time()
    ctl.ground([("base", [])])
    if verbose:
        print("Grounding took {}s".format(time.time() - grounding_start_time))

    if bound_all_events:
        ctl.ground([("bound_all_events", [int(bound_all_events)])])
    all_answers = _asp_loop(ctl, dl, verbose, debug)
    statistics: Dict = ctl.statistics

    if verbose:
        print(all_answers)
        _print_configuration(ctl)
        _print_stats(statistics)

    return FluxHelperResult(all_answers, statistics, ctl, dl)


def _asp_loop(ctl, dl, verbose, debug):
    all_answers = []
    min_cost = np.inf
    with ctl.solve(yield_=True) as handle:
        for model in handle:
            if len(model.cost) > 0:
                cost = model.cost[0]
                if cost < min_cost:
                    if verbose:
                        print("Optimization: {}".format(cost))
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
    return all_answers


def _print_stats(statistics):
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
    print("Time         : {:5.3f}s (Solving: {}s 1st Model: {}s Unsat: {}s)"
          .format(statistics["summary"]["times"]["total"],
                  statistics["summary"]["times"]["solve"],
                  statistics["summary"]["times"]["sat"],
                  statistics["summary"]["times"]["unsat"],
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