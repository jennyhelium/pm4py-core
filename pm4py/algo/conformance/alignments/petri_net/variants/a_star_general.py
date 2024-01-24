import heapq
import sys
import time
from copy import copy
from enum import Enum

import numpy as np

from pm4py.objects.log import obj as log_implementation
from pm4py.objects.petri_net.utils import align_utils as utils
from pm4py.objects.petri_net.utils.incidence_matrix import construct as inc_mat_construct
from pm4py.objects.petri_net.utils.synchronous_product import construct_cost_aware, construct
from pm4py.objects.petri_net.utils.petri_utils import construct_trace_net_cost_aware, decorate_places_preset_trans, \
    decorate_transitions_prepostset
from pm4py.util import exec_utils
from pm4py.util.constants import PARAMETER_CONSTANT_ACTIVITY_KEY
from pm4py.util.lp import solver as lp_solver
from pm4py.util.xes_constants import DEFAULT_NAME_KEY
from pm4py.util import variants_util
from typing import Optional, Dict, Any, Union
from pm4py.objects.log.obj import Trace
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.util import typing

"""
    Performs the basic alignment search on top of the synchronous product net, given a cost function and skip-symbol

    Parameters
    ----------
    sync_prod: :class:`pm4py.objects.petri.net.PetriNet` synchronous product net
    initial_marking: :class:`pm4py.objects.petri.net.Marking` initial marking in the synchronous product net
    final_marking: :class:`pm4py.objects.petri.net.Marking` final marking in the synchronous product net
    cost_function: :class:`dict` cost function mapping transitions to the synchronous product net
    skip: :class:`Any` symbol to use for skips in the alignment
    
    heuristic
    LP or ILP
"""


def search(sync_net, ini, fin, cost_function, skip, trace, activity_key, ret_tuple_as_trans_desc=False,
           max_align_time_trace=sys.maxsize, int_sol=False, solver=None):
    start_time = time.time()
    # create incidence matrix for sync net
    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)

    incidence_matrix = inc_mat_construct(sync_net)
    ini_vec, fin_vec, cost_vec = utils.__vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)

    a_matrix = np.asmatrix(incidence_matrix.a_matrix).astype(np.float64)
    g_matrix = -np.eye(len(sync_net.transitions))
    h_cvx = np.matrix(np.zeros(len(sync_net.transitions))).transpose()
    cost_vec = [x * 1.0 for x in cost_vec]

    use_cvxopt = False
    if lp_solver.DEFAULT_LP_SOLVER_VARIANT == lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN or lp_solver.DEFAULT_LP_SOLVER_VARIANT == lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP:
        use_cvxopt = True

    if use_cvxopt:
        # not available in the latest version of PM4Py
        from cvxopt import matrix

        a_matrix = matrix(a_matrix)
        g_matrix = matrix(g_matrix)
        h_cvx = matrix(h_cvx)
        cost_vec = matrix(cost_vec)

    # init set C "closed" which contains already visited markings
    closed = set()

    # compute heuristic for ini_state
    if int_sol:
        h, x = compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                       ini,
                                       fin_vec, lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP,
                                       use_cvxopt=use_cvxopt)
    else:
        h, x = compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                       ini,
                                       fin_vec, lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                       use_cvxopt=use_cvxopt)

    # Search Tupel (f = g+h, g bisherige Kosten, h Kosten bis final marking, p ??, t??, x?? trust??)
    ini_state = utils.SearchTuple(0 + h, 0, h, ini, None, None, x, True)

    # init priority queue Q, sorted ascending by f = g + h
    open_set = [ini_state]
    heapq.heapify(open_set)
    visited = 0
    queued = 0
    traversed = 0
    lp_solved = 1

    # ??
    trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

    # while Q not empty, remove its head and add to collection C "closed" (visited markings)
    while not len(open_set) == 0:
        if (time.time() - start_time) > max_align_time_trace:
            return None

        curr = heapq.heappop(open_set)

        current_marking = curr.m

        while not curr.trust:
            if (time.time() - start_time) > max_align_time_trace:
                return None

            # if marking already visited (in C), get new current marking from queue
            already_closed = current_marking in closed
            if already_closed:
                curr = heapq.heappop(open_set)
                current_marking = curr.m
                continue

            # else compute heuristic
            if int_sol:
                h, x = compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                               incidence_matrix, curr.m,
                                               fin_vec,
                                               lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP,
                                               use_cvxopt=use_cvxopt)
            else:
                h, x = compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                               incidence_matrix, curr.m,
                                               fin_vec,
                                               lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                               use_cvxopt=use_cvxopt)

            lp_solved += 1

            tp = utils.SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True)

            # push new SearchTuple in Q, pop next marking
            curr = heapq.heappushpop(open_set, tp)
            current_marking = curr.m

        # max allowed heuristics value (27/10/2019, due to the numerical instability of some of our solvers)
        if curr.h > lp_solver.MAX_ALLOWED_HEURISTICS:
            continue

        # 12/10/2019: do it again, since the marking could be changed
        already_closed = current_marking in closed
        if already_closed:
            continue

        # 12/10/2019: the current marking can be equal to the final marking only if the heuristics
        # (underestimation of the remaining cost) is 0. Low-hanging fruits
        if curr.h < 0.01:
            # if head represents final marking: return corresp. alignment by reconstructing
            if current_marking == fin:
                return utils.__reconstruct_alignment(curr, visited, queued, traversed,
                                                     ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                     lp_solved=lp_solved)

        closed.add(current_marking)
        visited += 1

        # fire each enabled transition and investigate new marking
        enabled_trans = copy(trans_empty_preset)
        for p in current_marking:
            for t in p.ass_trans:
                if t.sub_marking <= current_marking:
                    enabled_trans.add(t)

        trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans if not (
                t is not None and utils.__is_log_move(t, skip) and utils.__is_model_move(t, skip))]

        for t, cost in trans_to_visit_with_cost:
            traversed += 1
            new_marking = utils.add_markings(current_marking, t.add_marking)

            # if new marking in C, do nothing.
            if new_marking in closed:
                continue

            # if in Q, replace if lower path cost
            # if nothing exists in Q, insert with path cost and heuristic value
            g = curr.g + cost

            queued += 1
            h, x = utils.__derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
            trustable = utils.__trust_solution(x)
            new_f = g + h

            tp = utils.SearchTuple(new_f, g, h, new_marking, curr, t, x, trustable)
            heapq.heappush(open_set, tp)


def search_extended_marking_eq(sync_net, ini, fin, cost_function, skip, trace, activity_key, trace_net,
                               ret_tuple_as_trans_desc=False, max_align_time_trace=sys.maxsize, int_sol=False,
                               solver=None):
    # k-based underestimation
    # incrementally increase k
    # maximize reuse of previously computed solution vectors
    start_time = time.time()

    # create incidence matrix for sync net
    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)

    incidence_matrix = inc_mat_construct(sync_net)
    ini_vec, fin_vec, cost_vec = utils.__vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function)

    a_matrix = np.asmatrix(incidence_matrix.a_matrix).astype(np.float64)
    g_matrix = -np.eye(len(sync_net.transitions))
    h_cvx = np.matrix(np.zeros(len(sync_net.transitions))).transpose()
    cost_vec = [x * 1.0 for x in cost_vec]

    transitions_sorted = incidence_matrix.transitions

    use_cvxopt = False
    if lp_solver.DEFAULT_LP_SOLVER_VARIANT == lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN or lp_solver.DEFAULT_LP_SOLVER_VARIANT == lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP:
        use_cvxopt = True

    if use_cvxopt:
        # not available in the latest version of PM4Py
        from cvxopt import matrix

        a_matrix = matrix(a_matrix)
        g_matrix = matrix(g_matrix)
        h_cvx = matrix(h_cvx)
        cost_vec = matrix(cost_vec)

    # init set C "closed" which contains already visited markings
    closed = set()

    # init k for k-based underestimation
    k = 1

    # trace_string = [x[activity_key] for x in trace]
    # trace_division = [trace_string]
    trace_division = [trace]
    explained_events_list = [0]
    max_num_explained = 0

    # compute heuristic for ini_state
    # compute underestimate for k = 1, y_a refers to first transition in y_i starting with transition in trace model
    h, x, ilp_solved = compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                               ini,
                                               fin_vec, lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP,
                                               trace_division,
                                               use_cvxopt=use_cvxopt, heuristic="EXTENDED_STATE_EQUATION",
                                               int_sol=int_sol, k=k)

    # Search Tupel (f = g+h, g bisherige Kosten, h Kosten bis final marking, current marking is ini, p parent_state/marking, t ist transistion,wie state erreicht, x Lösung lp trust??)
    ini_state = utils.SearchTuple(0 + h, 0, h, ini, None, None, x, True)

    # init priority queue Q, sorted ascending by f = g + h
    open_set = [ini_state]
    heapq.heapify(open_set)
    visited = 0
    queued = 0
    traversed = 0
    lp_solved = 1 + ilp_solved

    # ??
    trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

    # while Q not empty, remove its head and add to collection C "closed" (visited markings)
    while not len(open_set) == 0:
        if (time.time() - start_time) > max_align_time_trace:
            return None

        curr = heapq.heappop(open_set)

        current_marking = curr.m

        # marking with unknown solution vector: original estimate corresp. to unrealizable firing sequence
        # try to improve underestimate function by increasing k and choose new way to split trace
        while not curr.trust:
            if (time.time() - start_time) > max_align_time_trace:
                return None

            # if marking already visited (in C), get new current marking from queue
            already_closed = current_marking in closed
            if already_closed:
                curr = heapq.heappop(open_set)
                current_marking = curr.m
                continue

            # else
            # if marking explains a events for k = 1,
            # restart procedure from scratch with k = 2 and sigma = sigma_1 + sigma_2 with |sigma_1| = a
            explained = explained_events(trace_net, current_marking)

            # number of newly explained events
            if explained > 0:
                new_explained = abs(max_num_explained - explained)
            else:
                new_explained = 0

            if explained > max_num_explained:
                max_num_explained = explained
            # if marking found which explains more events than before, increase k and split trace
            # else do not
            if new_explained > 0:
                explained_events_list.append(explained)

                split_element = trace_division.pop()

                if len(split_element) == 1:
                    trace_division.append(split_element)
                    k = k

                elif new_explained == len(split_element):
                    trace_division.append(split_element)
                    k = k

                else:
                    split_1 = split_element[:new_explained]
                    split_2 = split_element[new_explained:]

                    trace_division.append(split_1)

                    if not len(split_2) == 0:
                        trace_division.append(split_2)
                        # increase k
                        k = k + 1
                    else:
                        k = k

            # compute exact solution
            h, x, ilp_solved = compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                                       incidence_matrix, curr.m, fin_vec,
                                                       lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP,
                                                       trace_division,
                                                       use_cvxopt=use_cvxopt, heuristic="EXTENDED_STATE_EQUATION",
                                                       int_sol=int_sol, k=k)
            lp_solved = lp_solved + 1 + ilp_solved

            tp = utils.SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True)

            # push new SearchTuple in Q, pop next marking
            curr = heapq.heappushpop(open_set, tp)
            current_marking = curr.m

        # max allowed heuristics value (27/10/2019, due to the numerical instability of some of our solvers)
        if curr.h > lp_solver.MAX_ALLOWED_HEURISTICS:
            continue

        # 12/10/2019: do it again, since the marking could be changed
        already_closed = current_marking in closed
        if already_closed:
            continue

        # 12/10/2019: the current marking can be equal to the final marking only if the heuristics
        # (underestimation of the remaining cost) is 0. Low-hanging fruits
        if curr.h < 0.01:
            # if head represents final marking: return corresp. alignment by reconstructing
            if current_marking == fin:
                return utils.__reconstruct_alignment(curr, visited, queued, traversed,
                                                     ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                     lp_solved=lp_solved)

        closed.add(current_marking)
        visited += 1

        # fire each enabled transition and investigate new marking
        enabled_trans = copy(trans_empty_preset)
        for p in current_marking:
            for t in p.ass_trans:
                if t.sub_marking <= current_marking:
                    enabled_trans.add(t)

        trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans if not (
                t is not None and utils.__is_log_move(t, skip) and utils.__is_model_move(t, skip))]

        for t, cost in trans_to_visit_with_cost:
            traversed += 1
            new_marking = utils.add_markings(current_marking, t.add_marking)

            # if new marking in C, do nothing.
            if new_marking in closed:
                continue

            # if in Q, replace if lower path cost
            # if nothing exists in Q, insert with path cost and heuristic value
            g = curr.g + cost

            queued += 1
            h, x = utils.__derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
            trustable = utils.__trust_solution(x)
            new_f = g + h

            tp = utils.SearchTuple(new_f, g, h, new_marking, curr, t, x, trustable)
            heapq.heappush(open_set, tp)


def search_naive(sync_net, ini, fin, cost_function, skip, trace, activity_key, trace_net,
                 ret_tuple_as_trans_desc=False, max_align_time_trace=sys.maxsize, use_naive=True, solver=None):
    start_time = time.time()

    # create incidence matrix for sync net
    decorate_transitions_prepostset(sync_net)
    decorate_places_preset_trans(sync_net)

    # init set C "closed" which contains already visited markings
    closed = set()

    # compute heuristic for ini_state
    remaining_events = len(trace)

    h = 0
    if use_naive:
        for i in range(remaining_events):
            index = remaining_events - i
            label = trace[-index]
            h = h + compute_naive_heuristic(label, sync_net, cost_function)

    # Search Tupel (f = g+h, g bisherige Kosten, h Kosten bis final marking, current marking is ini, p parent_state/marking, t ist transistion,wie state erreicht, x Lösung lp trust??)
    ini_state = utils.SearchTuple(0 + h, 0, h, ini, None, None, None, True)

    # init priority queue Q, sorted ascending by f = g + h
    open_set = [ini_state]
    heapq.heapify(open_set)
    visited = 0
    queued = 0
    traversed = 0

    # ??
    trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

    # while Q not empty, remove its head and add to collection C "closed" (visited markings)
    while not len(open_set) == 0:
        if (time.time() - start_time) > max_align_time_trace:
            return None

        curr = heapq.heappop(open_set)

        current_marking = curr.m

        # marking with unknown solution vector: original estimate corresp. to unrealizable firing sequence
        while not curr.trust:
            if (time.time() - start_time) > max_align_time_trace:
                return None

            # if marking already visited (in C), get new current marking from queue
            already_closed = current_marking in closed
            if already_closed:
                curr = heapq.heappop(open_set)
                current_marking = curr.m
                continue

            # else
            h = 0

            if use_naive:
                # compute exact solution
                explained = explained_events(trace_net, current_marking)

                remaining_events = len(trace) - explained

                for i in range(remaining_events):
                    index = remaining_events - i
                    label = trace[-index]
                    h = h + compute_naive_heuristic(label, sync_net, cost_function)

            tp = utils.SearchTuple(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, None, True)

            # push new SearchTuple in Q, pop next marking
            curr = heapq.heappushpop(open_set, tp)
            current_marking = curr.m

        # max allowed heuristics value (27/10/2019, due to the numerical instability of some of our solvers)
        if curr.h > lp_solver.MAX_ALLOWED_HEURISTICS:
            continue

        # 12/10/2019: do it again, since the marking could be changed
        already_closed = current_marking in closed
        if already_closed:
            continue

        # 12/10/2019: the current marking can be equal to the final marking only if the heuristics
        # (underestimation of the remaining cost) is 0. Low-hanging fruits
        if curr.h < 0.01:
            # if head represents final marking: return corresp. alignment by reconstructing
            if current_marking == fin:
                current_h = curr.h
                return utils.__reconstruct_alignment(curr, visited, queued, traversed,
                                                     ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                     lp_solved=0)

        closed.add(current_marking)
        visited += 1

        # fire each enabled transition and investigate new marking
        enabled_trans = copy(trans_empty_preset)
        for p in current_marking:
            for t in p.ass_trans:
                if t.sub_marking <= current_marking:
                    enabled_trans.add(t)

        trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans if not (
                t is not None and utils.__is_log_move(t, skip) and utils.__is_model_move(t, skip))]

        for t, cost in trans_to_visit_with_cost:
            traversed += 1
            new_marking = utils.add_markings(current_marking, t.add_marking)

            # if new marking in C, do nothing.
            if new_marking in closed:
                continue

            # if in Q, replace if lower path cost
            # if nothing exists in Q, insert with path cost and heuristic value
            g = curr.g + cost

            queued += 1
            # h = len(trace) - explained_events(trace_net, new_marking)

            explained = explained_events(trace_net, current_marking)

            remaining_events = len(trace) - explained

            h = 0

            if use_naive:
                for i in range(remaining_events):
                    index = remaining_events - i
                    label = trace[-index]
                    naive = compute_naive_heuristic(label, sync_net, cost_function)
                    h = h + naive
            new_f = g + h

            tp = utils.SearchTuple(new_f, g, h, new_marking, curr, t, None, True)
            heapq.heappush(open_set, tp)


def compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                            marking, fin_vec, variant, trace_division=[], use_cvxopt=False, heuristic="STATE_EQUATION",
                            solver=None,
                            strict=True, int_sol=False, k=1):
    if heuristic == "STATE_EQUATION":
        return utils.__compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                                           incidence_matrix, marking, fin_vec,
                                                           variant,
                                                           use_cvxopt=use_cvxopt)
    elif heuristic == "EXTENDED_STATE_EQUATION":
        return utils.__compute_exact_extended_state_equation_ilp(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                                                 incidence_matrix, marking, fin_vec, variant,
                                                                 trace_division, ilp=int_sol,
                                                                 use_cvxopt=use_cvxopt, k=k)


def explained_events(trace_net, marking):
    """
    Return number of events corresponding to index in trace net already explained by given marking
    Parameters
    ----------
    trace_net
    marking

    Returns
    -------

    """
    trace_net_places = trace_net.places

    places = []

    for place in trace_net_places:
        places.append(place.name)

    places_int = [int(p[2:]) for p in places]

    places_sorted = sorted(places_int)

    for p in marking:
        curr_trace_place = p.name[0]

        if curr_trace_place in places:
            curr_trace_place_int = int(curr_trace_place[2:])

            if curr_trace_place_int in places_sorted:
                index = places_sorted.index(curr_trace_place_int)
                return index
    return 0


def compute_naive_heuristic(label, sync_net, cost_function):
    """
    Checks if there is a sync transition with label of trace
    If yes, return cost for sync move (0), else cost for log move (10000)
    Returns
    -------

    """

    # ((t_trace.name, t_model.name), (t_trace.label, t_model.label))
    for t in sync_net.transitions:
        if label == t.label[1]:
            # return cost_function[t]
            return 0
    return 10000
