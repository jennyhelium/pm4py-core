import pm4py
import pandas as pd
import numpy as np
from copy import copy
import random
from pm4py.objects.petri_net.utils import check_soundness
from pm4py.objects import petri_net


from statistics import mean


def random_playout(net, initial_marking, final_marking, no_traces, max_trace_length,
                   semantics=petri_net.semantics.ClassicSemantics()):

    visited = 0
    queued = 0
    traversed = 0
    deadlock = 0
    boundedness = 0

    all_visited_elements = []

    i = 0
    while True:
        if len(all_visited_elements) >= no_traces:
            break

        if i >= no_traces:
            if len(all_visited_elements) == 0:
                break

        visited_elements = []  # marking and transitions
        visible_transitions_visited = []

        marking = copy(initial_marking)
        while len(visible_transitions_visited) < max_trace_length:
            visited += 1
            visited_elements.append(marking)

            if not semantics.enabled_transitions(net, marking):  # deadlock
                deadlock += 1
                break

            if marking is not None:
                # check for boundedness
                if marking:
                    max_value = max(marking.values())
                    if max_value > boundedness:
                        boundedness = max_value

            all_enabled_trans = semantics.enabled_transitions(net, marking)
            queued += len(all_enabled_trans)

            if final_marking is not None and (final_marking == marking):  # final marking reached
                trans = random.choice(list(all_enabled_trans.union({None})))  # pick random enabled transition
            else:
                trans = random.choice(list(all_enabled_trans))

            if trans is None:
                break

            #traversed += 1
            visited_elements.append(trans)

            if trans.label is not None:
                visible_transitions_visited.append(trans)

            marking = semantics.execute(trans, net, marking)


        all_visited_elements.append(tuple(visited_elements))

        i += 1

    return all_visited_elements, visited, queued, deadlock, boundedness


def free_choice(net):
    for p in net.places:
        # check if outgoing arc only arc from that place
        if len(p.out_arcs) <= 1:
            continue
        else:  # check if outgoing arc is only arc to that transition
            for arc in p.out_arcs:
                curr_target = arc.target.name

                pre_set = []
                for t in net.arcs:  # set of arcs
                    if t.target.name != curr_target:
                        continue
                    else:
                        pre_set.append(t.source)
                if len(pre_set) > 1:
                    return 0
    return 1


def count_places(net):
    places = net.places
    return len(places)


def count_transitions(net):
    transitions = net.transitions
    return len(transitions)


def degree(net, use_place=True, direction="in"):
    """
    Degree of places/transitions
    Parameters
    ----------
    net
    use_place
    direction

    Returns
    -------

    """
    max_degree = 0
    min_degree = len(net.places)
    mean_degree = []

    if use_place:
        iterator = net.places
    else:
        iterator = net.transitions

    for i in iterator:

        if direction == "in":
            arcs = i.in_arcs
        else:
            arcs = i.out_arcs

        len_arcs = len(arcs)
        mean_degree.append(len_arcs)

        if len_arcs > max_degree:
            max_degree = len_arcs

        if len_arcs < min_degree:
            min_degree = len_arcs

    mean_degree = np.mean(mean_degree)

    return min_degree, max_degree, mean_degree


def degree_ratio(pn, use_place=True, direction="in"):
    if use_place:
        len_el = len(pn.places)
    else:
        len_el = len(pn.transitions)

    min_degree, max_degree, mean_degree = degree(pn, use_place, direction)

    return min_degree / len_el, max_degree / len_el, mean_degree / len_el


def choice(pn):
    sum_degree = 0
    mult_degree = 1

    for p in pn.places:
        len_arcs = len(p.out_arcs)

        if len_arcs > 0:
            sum_degree = sum_degree + len_arcs
            mult_degree = mult_degree * len_arcs

    ratio = sum_degree / len(pn.places)
    return sum_degree, ratio, mult_degree, mult_degree / len(pn.places)


def parallelism(pn):
    sum_degree = 0

    transitions = pn.transitions

    for t in transitions:
        len_arcs = len(t.out_arcs)
        sum_degree = sum_degree + len_arcs

    ratio = sum_degree / len(transitions)
    return sum_degree, ratio


def parallelism_model_multiplied(pn):
    mult_degree = 1

    transitions = pn.transitions

    for t in transitions:
        len_arcs = len(t.out_arcs)
        if len_arcs > 0:
            mult_degree *= len_arcs

    ratio = mult_degree / len(transitions)

    return mult_degree, ratio


def len_trace(trace):
    return len(trace)


def trace_ratio(pn, trace):
    # t = [x['concept:name'] for x in trace]
    t = trace
    len_trace = len(t)

    num_transitions = len(pn.transitions)
    num_places = len(pn.transitions)

    return len_trace, len_trace / num_transitions, len_trace / num_places, num_transitions / len_trace, num_places / len_trace


def countX(lst, x):
    return lst.count(x)


def model_silent_transitions(pn):
    transitions = pn.transitions

    labels = [t.label for t in transitions if t.label is None]

    silent_transitions = len(labels)

    ratio = silent_transitions / len(transitions)

    return silent_transitions, ratio


def model_duplicates(pn):
    transitions = pn.transitions

    labels = [t.label for t in transitions if t.label is not None]

    counts = []
    visited = []

    for t in labels:
        if t not in visited:
            count_t = countX(labels, t)
            counts.append(count_t)
            visited.append(t)

    # result = filter(lambda x: x > 1, counts)
    result = [x for x in counts if x > 1]

    num_duplicates = len(result)

    if len(labels) == 0:
        ratio = 0
    else:
        ratio = num_duplicates / len(labels)

    return num_duplicates, ratio


def trace_loop(trace):
    # t = [x['concept:name'] for x in trace]
    t = trace

    visited = []
    counts = []

    for i in t:
        if i not in visited:
            count_i = countX(t, i)
            counts.append(count_i)
            visited.append(i)

    # result = filter(lambda x: x > 1, counts)
    result = [x for x in counts if x > 1]

    num_repetitions = len(result)

    is_empty = False

    if len(result) == 0:
        is_empty = True

    if is_empty:
        max_repetitions = 0
        mean_repetitions = 0
        sum_repetitions = 0
    else:
        max_repetitions = max(result)
        mean_repetitions = mean(result)
        sum_repetitions = sum(result)

    ratio = num_repetitions / len(trace)

    return (num_repetitions, ratio, max_repetitions, max_repetitions / len(trace),
            mean_repetitions, mean_repetitions / len(trace), sum_repetitions, sum_repetitions / len(trace))

def one_length_loop(trace):
    t = trace

    prev = ""
    for i in t:
        if prev == i:
            return 1
        else:
            prev = i

    return 0


def distinct_events_trace(trace):
    distinct_events = []

    for t in trace:
        if t not in distinct_events:
            distinct_events.append(t)

    return len(distinct_events)

def transitions_no_in_arc(pn):
    transitions = pn.transitions

    count = 0

    for t in transitions:
        if len(t.in_arcs) == 0:
            count = count + 1

    return count, count / len(transitions)


def matching_loop(pn, trace):
    num_repetitions, _, _, _, _, _, _, _ = trace_loop(trace)

    trans_no_in_arc, _ = transitions_no_in_arc(pn)

    loop_trace = False
    loop_model = False

    if num_repetitions > 0:
        loop_trace = True

    if trans_no_in_arc > 0:
        loop_model = True

    return loop_model and loop_trace


def matching_labels(pn, trace):
    # t = [x['concept:name'] for x in trace]
    t = trace

    transitions = pn.transitions
    labels_model = [t.label for t in transitions if t.label is not None]

    match = []
    match_count_trace = 0

    match_count_model = 0

    for i in t:
        if i in labels_model:
            match_count_trace = match_count_trace + 1

    for i in labels_model:
        if i in t:
            match_count_model = match_count_model + 1

    if len(labels_model) == 0:
        ratio = 0
    else:
        ratio = match_count_model / len(labels_model)
    return match_count_trace, match_count_trace / len(t), match_count_model, ratio


def matching_starts(pn, trace):
    start_label = trace[0]

    for p in pn.places:
        if len(p.in_arcs) == 0:
            source = p

    start_transitions = []
    for arc in source.out_arcs:
        start_transitions.append(arc.target.label)

    if start_label in start_transitions:
        return 1

    return 0


def matching_ends(pn, trace):
    end_label = trace[-1]

    for p in pn.places:
        if len(p.out_arcs) == 0:
            sink = p

    end_transitions = []
    for arc in sink.in_arcs:
        end_transitions.append(arc.source.label)

    if end_label in end_transitions:
        return 1

    return 0



def token_based_reaply(trace, pn, im, fm):
    pm4py.conformance_diagnostics_token_based_replay()


if __name__ == "__main__":
    df_problems = pm4py.format_dataframe(pd.read_csv('pm4py/data/running_example_broken.csv', sep=';'),
                                        case_id='case:concept:name', activity_key='concept:name',
                                       timestamp_key='time:timestamp')
    pn, im, fm = pm4py.discover_petri_net_inductive(df_problems, noise_threshold=0)
    #road = pm4py.read_xes("pm4py/data/Road_Traffic_Fine_Management_Process.xes")
    #pn, im, fm = pm4py.discover_petri_net_inductive(road, noise_threshold=0)

    pm4py.view_petri_net(pn, im, fm)
    #print(free_choice(pn))
    #visited_elements, visited, queued, deadlock, bound = random_playout(pn, im, fm, 5, 50)
    #print(visited_elements)
    #print(queued)
    #print(visited)
    #print(deadlock)
    #print(bound)

    trace = ['examine thoroughly', 'check ticket', 'examine thoroughly', 'decide', 'reject request']
    print(one_length_loop(trace))
    print(distinct_events_trace(trace))
    print(matching_starts(pn, trace))
    print(matching_ends(pn, trace))
"""
    print(degree(pn))
    print(degree(pn, direction="out"))
    print(degree(pn, False))
    print(degree(pn, False, "out"))

    print(degree_ratio(pn))
    print(choice(pn))
    print(parallelism(pn))
    print(trace_ratio(pn, trace))
    print(model_silent_transitions(pn))
    print(model_duplicates(pn))
    print(transitions_no_in_arc(pn))
    print(trace_loop(trace))
    print(matching_loop(pn, trace))
    print(matching_labels(pn, trace))
    print(parallelism_model_multiplied(pn))
    print(choice(pn))
"""

