import networkx as nx
from rsp.schedule_problem_description.data_types_and_utils import RouteDAGConstraints
from rsp.schedule_problem_description.data_types_and_utils import schedule_problem_description_equals
from rsp.schedule_problem_description.data_types_and_utils import ScheduleProblemDescription


def test_schedule_problem_description_equals():
    def _data() -> ScheduleProblemDescription:
        topo = nx.DiGraph()
        topo.add_edge(5, 10)
        topo.add_edge(6, 11)
        return ScheduleProblemDescription(
            route_dag_constraints_dict={0: RouteDAGConstraints(
                earliest={},
                latest={}
            )},
            topo_dict={0: topo},
            minimum_travel_time_dict={0: 1},
            max_episode_steps=555,
            route_section_penalties={0: {}},
            weight_lateness_seconds=1
        )

    s1 = _data()
    s2 = _data()
    assert s1 != s2
    assert schedule_problem_description_equals(s1, s2)
    assert schedule_problem_description_equals(s1, s1)
    s2.topo_dict[0].add_edge(33, 55)
    assert s1 != s2
    assert not schedule_problem_description_equals(s1, s2)
    assert schedule_problem_description_equals(s2, s2)
