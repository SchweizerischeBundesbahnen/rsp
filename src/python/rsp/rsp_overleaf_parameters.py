# noqa
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ExperimentAgenda
from rsp.step_01_agenda_expansion.experiment_parameters_and_ranges import ExperimentParameters
from rsp.utils.pickle_helper import _pickle_load


def _make_hashable(obj):
    if str(type(obj)) == "<class 'dict'>":
        return str(obj)
    return obj


if __name__ == "__main__":

    folder = "../rsp-data/PUBLICATION_DATA/PUBLICATION_DATA_baseline_2020_11_17T23_40_54"
    print(f"{folder}/experiment_agenda.pkl")
    for infra_id in range(48):
        ip = _pickle_load(folder=f"{folder}/../infra/{infra_id:03d}", file_name=f"infrastructure_parameters.pkl")
        print(ip)
    agenda: ExperimentAgenda = _pickle_load(folder=folder, file_name=f"experiment_agenda.pkl")

    print(type(agenda))
    print("=================================")
    print("global constants")
    print("=================================")
    print(agenda.global_constants)
    experiment0: ExperimentParameters = agenda.experiments[0]
    print("=================================")
    print("global constants")
    print("=================================")
    for item in ["experiment_id", "grid_id", "infra_id_schedule_id"]:
        vals = list(set([_make_hashable(experiment._asdict()[item]) for experiment in agenda.experiments]))
        print(f"{item} ({len(vals)}): {vals}")

    print("=================================")
    print("B.1 infra_parameters")
    print("=================================")
    for item in experiment0.infra_parameters._asdict().keys():
        vals = list(set([_make_hashable(experiment.infra_parameters._asdict()[item]) for experiment in agenda.experiments]))
        print(f"{item} ({len(vals)}): {vals}")
    print("=================================")
    print("B.2 schedule_parameters")
    print("=================================")
    for item in experiment0.schedule_parameters._asdict().keys():
        vals = list(set([_make_hashable(experiment.schedule_parameters._asdict()[item]) for experiment in agenda.experiments]))
        print(f"{item} ({len(vals)}): {vals}")
    print("=================================")
    print("B.3 re_schedule_parameters")
    print("=================================")
    for item in experiment0.re_schedule_parameters._asdict().keys():
        vals = list(set([_make_hashable(experiment.re_schedule_parameters._asdict()[item]) for experiment in agenda.experiments]))
        print(f"{item} ({len(vals)}): {vals}")
