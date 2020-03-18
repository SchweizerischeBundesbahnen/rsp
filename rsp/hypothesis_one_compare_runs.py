from rsp.hypothesis_testing.compare_runtimes import compare_runtimes

if __name__ == '__main__':
    # allow for non-matching pkl files
    # this should be safe here since we only consider solve times and solution costs
    COMPATIBILITY_MODE = False
    compare_runtimes(
        data_folder1='./exp_hypothesis_one_2020_03_04T19_19_00',
        data_folder2='./exp_hypothesis_one_2020_03_10T22_10_19',
        output_enclosing_folder='.',
        experiment_ids=[]
    )
