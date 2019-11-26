from flatland.envs.rail_trainrun_data_structures import Waypoint


def make_variable_name_agent_at_waypoint(agent_id, wp: Waypoint) -> str:
    """Returns a variable name for the event of an agent passing a waypoint."""
    return 'Var_Agent_ID_{}_at_{}_{}_{}_entry'.format(agent_id, *wp.position, wp.direction)
