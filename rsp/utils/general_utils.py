import pickle
import time


def current_milli_time():
    return int(round(time.time() * 1000))


VERIFICATION = False


def verification_by_file(title, actual, loop_index, solver_name, generate: bool = False, debug: bool = False):
    """Verifies that an actual against an expected object from a file. In
    generate mode, the value is written to file and re-read and verified to
    ensure that the verification_by_file of works correctly for the expected
    itself.

    Parameters
    ----------
    title
        label of what we want to check
    actual
        actual value
    loop_index
        for distinguishing labels of the same title
    solver_name
        for distinguishing solvers
    generate
        write actual value to file
    debug
        print actual and expected value

    Returns
    -------
    """
    if not VERIFICATION:
        return
    file_name = "{}_{}_expected{}.txt".format(loop_index, solver_name, title)
    if generate:
        with open(file_name, "wb") as file:
            expected_action_plan_string = pickle.dumps(actual)
            file.write(expected_action_plan_string)
    with open(file_name, "rb") as file:
        expected_string = file.read()
        expected = pickle.loads(expected_string)
        if debug:
            print("[{}] actual {}=\n{}".format(loop_index, title, actual))
            print("[{}] expected {}=\n{}".format(loop_index, title, expected))
        assert actual == expected, format("at {}".format(loop_index))
