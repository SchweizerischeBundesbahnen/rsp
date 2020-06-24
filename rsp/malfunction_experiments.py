from rsp.hypothesis_one_experiments import hypothesis_one_malfunction_analysis

if __name__ == '__main__':
    # do not commit your own calls !
    experiment_name = 'exp_hypothesis_one_2020_06_24T15_14_47'
    experiment_base_directory = './{}/'.format(experiment_name)

    experiment_base_directory = None
    hypothesis_one_malfunction_analysis(agenda_folder=experiment_base_directory,
                                        experiment_name=experiment_name,
                                        malfunction_agent_id=5)
