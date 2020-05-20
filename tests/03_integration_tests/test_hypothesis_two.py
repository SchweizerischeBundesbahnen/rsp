from rsp.hypothesis_two_encounter_graph import hypothesis_two_encounter_graph_directed


def test_hypothesis_two():
    """Run hypothesis two."""
    hypothesis_two_encounter_graph_directed(
        experiment_base_directory='./res/mini_toy_example',
        experiment_ids=[0],
        show=False
    )
