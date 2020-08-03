from pm4py.algo.conformance.alignments.algorithm import apply as get_alignment
from pm4py.algo.conformance.alignments.algorithm import Parameters
from pm4py.algo.conformance.alignments.algorithm import Variants
from pm4py.objects.conversion.process_tree.converter import apply as convert_pt_to_petri_net
from pm4py.objects.conversion.process_tree.converter import Variants as pt_petri_net_variants
from pm4py.objects.log.log import Trace, Event
from pm4py.objects.process_tree.process_tree import ProcessTree
from typing import Set, List
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.objects.petri.align_utils import SKIP, STD_MODEL_LOG_MOVE_COST


def concatenate_traces(t1: Trace, t2: Trace) -> Trace:
    for e in t2:
        t1.append(e)
    return t1


def trace_to_list_of_str(t: Trace) -> List[str]:
    return [e[DEFAULT_NAME_KEY] for e in t]


def list_of_str_to_trace(activities: List[str]) -> Trace:
    t = Trace()
    for a in activities:
        e = Event()
        e["concept:name"] = a
        t.append(e)
    return t


def get_activity_labels_from_trace(trace: Trace) -> Set[str]:
    labels = set()
    for a in trace:
        labels.add(a[DEFAULT_NAME_KEY])
    return labels


def apply_standard_cost_function_to_alignment(align: List) -> int:
    res = 0
    for move in align:
        if move[0] == SKIP and move[1] is not None:
            # visible model move
            res += 1
        elif move[0] != SKIP and move[1] == SKIP:
            # log move
            res += 1
    return res


def empty_sequence_accepted(pt: ProcessTree) -> bool:
    alignment = calculate_optimal_alignment(pt, Trace(), variant=Variants.VERSION_STATE_EQUATION_A_STAR)
    return alignment["cost"] < STD_MODEL_LOG_MOVE_COST


def calculate_optimal_alignment(pt: ProcessTree, trace: Trace, variant=Variants.VERSION_STATE_EQUATION_A_STAR,
                                parameters=None):
    parent = pt.parent
    pt.parent = None
    net, im, fm = convert_pt_to_petri_net(pt, variant=pt_petri_net_variants.TO_PETRI_NET_TRANSITION_BORDERED)
    alignment = get_alignment(trace, net, im, fm, variant=variant,
                              parameters={Parameters.PARAM_ALIGNMENT_RESULT_IS_SYNC_PROD_AWARE: True})
    pt.parent = parent
    # remove invisible model moves from alignment steps that do not belong to a silent model move in the process tree
    res = []
    for a in alignment["alignment"]:
        if not (a[0][0] == SKIP and not a[0][1].isdigit()):
            res.append(a[1])
    return res
