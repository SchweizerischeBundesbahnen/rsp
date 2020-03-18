from rsp.hypothesis_testing.compare_runtimes import compare_runtimes

if __name__ == '__main__':
    # allow for non-matching pkl files
    # this should be safe here since we only consider solve times and solution costs
    COMPATIBILITY_MODE = True
    compare_runtimes(
        data_folder1='./hypothesis_testing/exp_006_hypothesis_window_size_null_2020_03_18T08_32_25/data',
        data_folder2='./hypothesis_testing/exp_006_hypothesis_window_size_alt000_2020_03_18T11_48_59/data',
        output_enclosing_folder='.',
        experiment_ids=[],
        fail_on_missing_experiment_ids=True
    )
