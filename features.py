import pm4py
import pandas as pd
import numpy as np

from statistics import mean

df_problems = pm4py.format_dataframe(pd.read_csv('pm4py/data/running_example_broken.csv', sep=';'),
                                     case_id='case:concept:name', activity_key='concept:name',
                                     timestamp_key='time:timestamp')
pn, im, fm = pm4py.discover_petri_net_inductive(df_problems)

trace = ['examine thoroughly', 'check ticket', 'examine thoroughly', 'decide', 'reject request']


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

    for p in pn.places:
        len_arcs = len(p.out_arcs)
        sum_degree = sum_degree + len_arcs

    ratio = sum_degree / len(pn.places)
    return sum_degree, ratio


def parallelism(pn):
    sum_degree = 0

    transitions = pn.transitions

    for t in transitions:
        len_arcs = len(t.out_arcs)
        sum_degree = sum_degree + len_arcs

    ratio = sum_degree / len(transitions)
    return sum_degree, ratio


def trace_ratio(pn, trace):
    #t = [x['concept:name'] for x in trace]
    t = trace
    len_trace = len(t)

    num_transitions = len(pn.transitions)
    num_places = len(pn.transitions)

    return len_trace, len_trace / num_transitions, len_trace / num_places


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

    #result = filter(lambda x: x > 1, counts)
    result = [x for x in counts if x > 1]

    num_duplicates = len(result)

    ratio = num_duplicates / len(labels)

    return num_duplicates, ratio


def trace_loop(trace):
    #t = [x['concept:name'] for x in trace]
    t = trace

    visited = []
    counts = []

    for i in t:
        if i not in visited:
            count_i = countX(t, i)
            counts.append(count_i)
            visited.append(i)

    #result = filter(lambda x: x > 1, counts)
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
    #t = [x['concept:name'] for x in trace]
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

    return match_count_trace, match_count_trace / len(t), match_count_model, match_count_model / len(labels_model)



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