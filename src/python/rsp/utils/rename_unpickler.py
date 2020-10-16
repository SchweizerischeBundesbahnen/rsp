import pickle

MODULE_RENAME_MAPPING = {
    ("rsp.experiment_solvers.data_types", "Infrastructure"): ("rsp.step_02_setup.data_types", "Infrastructure"),
    ("rsp.experiment_solvers.data_types", "Schedule"): ("rsp.scheduling.schedule", "Schedule"),
    ("rsp.experiment_solvers.data_types", "SchedulingExperimentResult"): ("rsp.scheduling.schedule", "SchedulingExperimentResult"),
    ("rsp.schedule_problem_description.data_types_and_utils", "ScheduleProblemDescription"): (
        "rsp.scheduling.scheduling_problem",
        "ScheduleProblemDescription",
    ),
    ("rsp.schedule_problem_description.data_types_and_utils", "TopoDict"): ("rsp.scheduling.scheduling_problem", "TopoDict"),
    ("rsp.schedule_problem_description.data_types_and_utils", "AgentPaths"): ("rsp.scheduling.scheduling_problem", "AgentPaths"),
    ("rsp.schedule_problem_description.data_types_and_utils", "AgentsPathsDict"): ("rsp.scheduling.scheduling_problem", "AgentsPathsDict"),
    ("rsp.schedule_problem_description.data_types_and_utils", "RouteDAGConstraints"): ("rsp.scheduling.scheduling_problem", "RouteDAGConstraints"),
    ("rsp.schedule_problem_description.data_types_and_utils", "RouteDAGConstraintsDict"): ("rsp.scheduling.scheduling_problem", "RouteDAGConstraintsDict"),
    ("rsp.schedule_problem_description.data_types_and_utils", "RouteDagEdge"): ("rsp.scheduling.scheduling_problem", "RouteDagEdge"),
    ("rsp.schedule_problem_description.data_types_and_utils", "RouteSectionPenalties"): ("rsp.scheduling.scheduling_problem", "RouteSectionPenalties"),
    ("rsp.schedule_problem_description.data_types_and_utils", "WaypointPenalties"): ("rsp.scheduling.scheduling_problem", "WaypointPenalties"),
    ("rsp.schedule_problem_description.data_types_and_utils", "RouteSectionPenaltiesDict"): ("rsp.scheduling.scheduling_problem", "RouteSectionPenaltiesDict"),
}


# https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        renamed_module, renamed_name = MODULE_RENAME_MAPPING.get((module, name), (module, name))
        return super(RenameUnpickler, self).find_class(renamed_module, renamed_name)
