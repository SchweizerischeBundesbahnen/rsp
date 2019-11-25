"""Run scheduling problem with 1000 generated scenarios."""
# ----------------------------- Helper -----------------------------------------------------
from solver.solve_utils import test_helper


# main function
def main():
    rendering = True
    nbr_of_tests = 1000

    output_file_name = "solve_tests.csv"
    tests = range(nbr_of_tests)

    test_helper(output_file_name, rendering, tests)


if __name__ == '__main__':
    main()
