from rsp.hypothesis_one_experiments import hypothesis_one_malfunction_analysis

if __name__ == '__main__':
    # do not commit your own calls !
    experiment_base_directory = '../rsp-data/agent_0_malfunction_2020_06_22T11_48_47/'
    experiment_base_directory = None
    hypothesis_one_malfunction_analysis(agenda_folder=experiment_base_directory,
                                        malfunction_agent_id=5)
