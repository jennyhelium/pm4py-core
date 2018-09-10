import re
from pm4py.models import petri
from pm4py.models.transition_system import transition_system as ts
from pm4py.models.transition_system import utils

def staterep(name):
    '''
        Creates a string representation for a state of a transition system.

        Parameters
        ----------
        name: the name of a state

        Returns
        -------
        Version of the name filtered of non-alphanumerical characters (except '_').
    '''
    return re.sub(r'\W+', '', name)

def construct_reachability_graph(net, initial_marking):
    '''
        Creates a reachability graph of a certain Petri net.
        DO NOT ATTEMPT WITH AN UNBOUNDED PETRI NET, EVER.

        Parameters
        ----------
        net: Petri net
        initial_marking: initial marking of the Petri net.

        Returns
        -------
        re_gr: Transition system that represents the reachability graph of the input Petri net.
    '''
    active = [initial_marking]
    visited = []
    re_gr = ts.TransitionSystem()
    re_gr.states.add(ts.TransitionSystem.State(staterep(repr(initial_marking))))
    while active:
        curr_mark = active.pop(0)
        curr_state = next((state for state in re_gr.states if state.name == staterep(repr(curr_mark))), None)
        en_tr = petri.semantics.enabled_transitions(net, curr_mark)
        for t in en_tr:
            next_mark = petri.semantics.execute(t, net, curr_mark)
            next_state = next((state for state in re_gr.states if state.name == staterep(repr(next_mark))), None)
            if next_state is None:
                next_state = ts.TransitionSystem.State(staterep(repr(next_mark)))
                re_gr.states.add(next_state)
            utils.add_arc_from_to(repr(t), curr_state, next_state, re_gr)
            if hash(next_mark) not in visited and next((mark for mark in active if hash(mark) == hash(next_mark)), None) is None and hash(curr_mark) != hash(next_mark):
                active.append(next_mark)
        visited.append(hash(curr_mark))
    return re_gr


if __name__ == '__main__':
    from pm4py.models.petri.importer import pnml as petri_importer
    from pm4py.models.transition_system import visualize as ts_vis

    net, initial_marking = petri_importer.import_petri_from_pnml(
        r'C:\Users\pegoraro\Google Drive\PADS\uncertainty\alignments\test1\net1.pnml')
    ts = construct_reachability_graph(net, initial_marking)
    vis = ts_vis.graphviz.visualize(ts)
    vis.view()
