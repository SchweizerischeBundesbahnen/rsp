from rsp.hypothesis_testing.run_null_alt_agenda import compare_agendas
from rsp.hypothesis_testing.tweak_experiment_agenda import tweak_asp_seed_value
from rsp.hypothesis_testing.tweak_experiment_agenda import tweak_name
from rsp.utils.experiments import load_experiment_agenda_from_file


def hypothesis_001_solver_seeding_irrelevant_main(copy_agenda_from_base_directory: str):
    """Copy agenda (A.1) and schedule/malfunction (A.2) and run pipeline from B
    multiple times with tweaked agenda and compare the runs D.

    Parameters
    ----------
    copy_agenda_from_base_directory
    """
    experiment_name = "plausi_001"
    agenda_null = tweak_name(
        agenda_null=load_experiment_agenda_from_file(copy_agenda_from_base_directory),
        alt_index=None,
        experiment_name=experiment_name)
    compare_agendas(
        experiment_agenda_null=agenda_null,
        experiment_name=experiment_name,
        experiment_agenda_alternatives=[
            tweak_asp_seed_value(
                agenda_null=agenda_null, seed=(94 + inc),
                alt_index=inc,
                experiment_name=experiment_name)
            for inc in range(5)],
        copy_agenda_from_base_directory=copy_agenda_from_base_directory
    )


if __name__ == '__main__':
    hypothesis_001_solver_seeding_irrelevant_main(
        copy_agenda_from_base_directory='exp_hypothesis_one_2020_03_31T07_11_03')
