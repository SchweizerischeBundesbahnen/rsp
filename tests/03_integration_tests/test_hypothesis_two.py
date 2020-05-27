from rsp.hypothesis_two_encounter_graph import hypothesis_two_disturbance_propagation_graph


def test_hypothesis_two():
    """Run hypothesis two."""
    hypothesis_two_disturbance_propagation_graph(
        experiment_base_directory='./res/mini_toy_example',
        experiment_ids=[0],
        show=False
    )
