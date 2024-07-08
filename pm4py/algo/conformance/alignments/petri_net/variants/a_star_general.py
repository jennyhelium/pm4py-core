import heapq
import sys
import time
from timeit import default_timer as timer
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

    _search_timer = timer()
    lp_time = 0.0
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

    num_real_sol = 0
    # compute heuristic for ini_state
    _lp_timer = timer()
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
        if not utils.check_lp_sol_int(x):
            num_real_sol = num_real_sol + 1
    lp_time += timer() - _lp_timer

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
            _lp_timer = timer()
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
                if not utils.check_lp_sol_int(x):
                    num_real_sol = num_real_sol + 1
            lp_time += timer() - _lp_timer
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
                                               use_cvxopt=True, heuristic="EXTENDED_STATE_EQUATION",
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
                                                       use_cvxopt=True, heuristic="EXTENDED_STATE_EQUATION",
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


def search_extended_marking_eq_correct(sync_net, ini, fin, cost_function, skip, trace, activity_key, trace_net,
                               ret_tuple_as_trans_desc=False, max_align_time_trace=sys.maxsize, int_sol=False,
                               solver=None):
    start_time = timer()
    lp_time = 0.0
    ext_state_eq_time = 0.0
    verbose = False

    preprocessing_time = 0.0
    trace_splitting_time = 0.0
    queue_pop_time = 0.0
    queue_insertion_time = 0.0
    queue_update_time = 0.0
    closed_set_time = 0.0
    graph_nav_time = 0.0
    
    if verbose:
        print("Starting Ext State Eq A* search with trace len", len(trace))

    _preprocessing_timer = timer()
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

    # init k for k-based underestimation
    split_points = []
    lp_solved = 0

    solved_extended_state_eqs = 0
    solved_state_eqs = 0

    # init table of computed state equations
    computed_exact_heuristics = {}

    preprocessing_time = timer() - _preprocessing_timer

    round_1 = True

    while True:  # no solution
        # init set C "closed" which contains already visited markings
        closed = set()

        _split_timer = timer()
        split_trace_ls = split_trace(trace, split_points)
        trace_splitting_time += timer() - _split_timer


        _lp_timer_start = timer()

        h, x = compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                    incidence_matrix, ini,
                                    fin_vec,
                                    lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP if int_sol else lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                    split_trace_ls, use_cvxopt=True, heuristic="EXTENDED_STATE_EQUATION",
                                    int_sol=True, k=len(split_points) + 1)
        
        if not round_1:
            if verbose:
                print("solution vector difference:", np.sum(np.abs(np.array(x) - np.array(old_start_x))))
        else:
            round_1 = False

        old_start_x = x

        if verbose: 
            print("Solution for current ext state eq is:", (h, np.sum(np.array(x))))
        elapsed = timer() - _lp_timer_start
        lp_time += elapsed
        ext_state_eq_time += elapsed
        solved_extended_state_eqs += 1

        ini_state = utils.SearchTupleExtStateEq(0 + h, 0, h, ini, None, None, x, True, 0)
        # init priority queue Q, sorted ascending by f = g + h
        open_set = [ini_state]
        open_set_custom_queue = utils.MappedQueue(open_set)

        visited = 0
        queued = 0
        traversed = 0
        lp_solved += 1

        num_explained_events = 0

        trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

        while not len(open_set_custom_queue) == 0:
            #if (time.time() - start_time) > max_align_time_trace:
             #   return None

            _pop_timer = timer()
            curr = open_set_custom_queue.pop()
            queue_pop_time += timer() - _pop_timer

            current_marking = curr.m

            # current marking can be equal to the final marking only if the heuristics
            # (underestimation of the remaining cost) is 0. Low-hanging fruits
            if curr.h < 0.01:
                # if head represents final marking: return corresp. alignment by reconstructing
                if current_marking == fin:
                    tot_time = timer() - start_time
                    total_queue_time = queue_pop_time + queue_insertion_time + queue_update_time
                    if verbose:
                        print("#####################################################################################")
                        print("TOTAL TIME:", tot_time)
                        print("lp solving time:", lp_time, "("+ str(100 * lp_time / tot_time) +"%)", "(" + str(solved_state_eqs) + " state / "+ str(solved_extended_state_eqs)+ " extended / " + str(lp_solved)+ " total)")
                        print("trace split time:", trace_splitting_time, "("+ str(100 * trace_splitting_time / tot_time) +"%)")
                        # print("queue push time:", queue_insertion_time, "("+ str(100 * queue_insertion_time / tot_time) +"%)")
                        print("queue update time:", queue_update_time, "("+ str(100 * queue_update_time / tot_time) +"%)")
                        # print("queue pop time:", queue_pop_time, "("+ str(100 * queue_pop_time / tot_time) +"%)")
                        print("total queue time:", total_queue_time, "("+ str(100 * total_queue_time / tot_time) +"%)")
                        print("preprocess time:", preprocessing_time, "("+ str(100 * preprocessing_time / tot_time) +"%)")
                        print("closed set time:", closed_set_time, "("+ str(100 * closed_set_time / tot_time) +"%)")
                        print("graph navigation time:", graph_nav_time, "("+ str(100 * graph_nav_time / tot_time) +"%)")
                        print("Final k was:", len(split_points) + 1, "trace length:", len(trace), "markings closed:", len(closed))
                        print("#####################################################################################")

                    # print("EXT STATE EQ FOUND SOLUTION WITH COST:", curr.g, "LEN ALIGNMENT:", len(utils.__reconstruct_alignment(curr, visited, queued, traversed,
                    #                                      ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                    #                                      lp_solved=lp_solved)["alignment"]))
                    return utils.__reconstruct_alignment(curr, visited, queued, traversed,
                                                         ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                         lp_solved=lp_solved)
            else:
                assert not current_marking == fin


            # 12/10/2019: do it again, since the marking could be changed
            _closed_set_timer = timer()
            already_closed = current_marking in closed
            if already_closed:
                continue
            closed_set_time += timer() - _closed_set_timer
                
            if not curr.trust:
                
                # print("First non - trusted marking seen!")
                # print("Current marking:", (curr.trust, curr.f, curr.g, curr.h))
                # print("MARKING QUEUE:", [(qe.trust, qe.f, qe.g, qe.h) for qe in sorted(open_set_custom_queue.h)])
                # assert False

                # restart search with longer list of split points
                if num_explained_events not in split_points and num_explained_events > 0 and num_explained_events < len(trace):
                    split_points.append(num_explained_events)
                    split_points = sorted(split_points)
                    # print("current search stopped. Solved", solved_state_eqs, "state equations and", solved_extended_state_eqs, "extended ones.")
                    # print("Time for ext state eq:", ext_state_eq_time, "Time for all lps:", lp_time, "Time for all:", timer() - start_time)
                    if verbose:
                        print("Restarting search with new splitpoints:", split_points)
                        print("Solved LPs so far:", lp_solved, "extended:", solved_extended_state_eqs)
                    break

                # compute true estimate
                _split_timer = timer()
                curr_explained = curr.num_explained
                subset_splitpoints = sorted([split_idx for split_idx in split_points if split_idx >= curr_explained])
                subset_split_trace = split_trace(trace, subset_splitpoints)[1:] #remove sigma_0
                trace_splitting_time += timer() - _split_timer

                # print("trace:", trace)
                # print("current splitpoints:", split_points)
                # print("current splitted trace:", split_trace_ls)
                # print("curr marking explained:", curr_explained)
                # print("subset splitpoints:", subset_splitpoints)
                # print("subset split trace:", subset_split_trace)

                _lp_timer_start = timer()
                if len(subset_split_trace) != 0:
                    h, x = compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                            incidence_matrix, curr.m,
                                            fin_vec,
                                            lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP if int_sol else lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                            subset_split_trace,
                                            use_cvxopt=use_cvxopt,
                                            heuristic = "EXTENDED_STATE_EQUATION",
                                            k = len(subset_split_trace))
                    solved_extended_state_eqs += 1
                else:
                    h, x = compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                                incidence_matrix, curr.m,
                                                fin_vec,
                                                lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP if int_sol else lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                                use_cvxopt=use_cvxopt)
                    solved_state_eqs += 1

                # print("had to compute exact sol for unknown :(")

                lp_time += timer() - _lp_timer_start
                lp_solved += 1

                tp = utils.SearchTupleExtStateEq(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True, curr.num_explained)
                _push_timer = timer()
                added_back = open_set_custom_queue.push(tp)
                queue_insertion_time += timer() - _push_timer
                assert added_back
                continue

            # max allowed heuristics value (27/10/2019, due to the numerical instability of some of our solvers)
            if curr.h > lp_solver.MAX_ALLOWED_HEURISTICS:
                continue

            _closed_set_timer = timer()
            closed.add(current_marking)
            visited += 1
            closed_set_time += timer() - _closed_set_timer

            num_explained_events = max(num_explained_events, curr.num_explained) # how do we determine explained events?

            _graph_nav_timer = timer()
            # fire each enabled transition and investigate new marking
            enabled_trans = copy(trans_empty_preset)
            for p in current_marking:
                for t in p.ass_trans:
                    if t.sub_marking <= current_marking:
                        enabled_trans.add(t)
        
            trans_to_visit_with_cost = [(t, cost_function[t]) for t in enabled_trans if not (
                    t is not None and utils.__is_log_move(t, skip) and utils.__is_model_move(t, skip))]
            graph_nav_time += timer() - _graph_nav_timer

            for t, cost in trans_to_visit_with_cost:
                traversed += 1
                new_marking = utils.add_markings(current_marking, t.add_marking)

                # if new marking in C, do nothing.
                _closed_set_timer = timer()
                if new_marking in closed:
                    continue
                closed_set_time += timer() - _closed_set_timer

                # cost to get to the new marking
                g = curr.g + cost

                # heuristic with the new path
                h, x = utils.__derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
                trustable = utils.__trust_solution(x)
                
                # print("Explored new marking - curr f =", curr.f, (curr.g, curr.h), "next f =", new_f, (g, h), "trusted?", trustable)

                # update num explained
                new_num_explained = curr.num_explained
                if curr.num_explained < len(trace):
                    next_label_to_explain = trace[curr.num_explained]
                    if t.label[0] == next_label_to_explain:              # what is trace net, wat is model net?
                        new_num_explained += 1

                # if in Q, replace if lower path cost
                # if nothing exists in Q, insert with path cost and heuristic value

                if new_marking not in open_set_custom_queue.d:
                    new_f = g + h
                    tp = utils.SearchTupleExtStateEq(new_f, g, h, new_marking, curr, t, x, trustable, new_num_explained)
                    _push_timer = timer()
                    added_back = open_set_custom_queue.push(tp)
                    queue_insertion_time += timer() - _push_timer
                    queued += 1
                elif g < open_set_custom_queue.h[open_set_custom_queue.d[new_marking]].g:
                    # re-use exact heuristics
                    if not trustable and open_set_custom_queue.h[open_set_custom_queue.d[new_marking]].trust:
                        h = open_set_custom_queue.h[open_set_custom_queue.d[new_marking]].h
                        x = open_set_custom_queue.h[open_set_custom_queue.d[new_marking]].x
                        trustable = True
                    new_f = g + h
                    tp = utils.SearchTupleExtStateEq(new_f, g, h, new_marking, curr, t, x, trustable, new_num_explained)
                    _queue_update_timer = timer()
                    open_set_custom_queue.update(open_set_custom_queue.h[open_set_custom_queue.d[new_marking]], tp)
                    queue_update_time += timer() - _queue_update_timer

        # if len(split_points) + 1 >= len(trace):
        #     print("THIS SHOULD ONLY HAPPEN ONCE?")


def search_extended_marking_eq_nodebug(sync_net, ini, fin, cost_function, skip, trace, activity_key, trace_net,
                               ret_tuple_as_trans_desc=False, max_align_time_trace=sys.maxsize, int_sol=False,
                               solver=None):
    start_time = timer()

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

    # init k for k-based underestimation
    split_points = []
    lp_solved = 0

    while True:  # no solution
        # init set C "closed" which contains already visited markings
        closed = set()

        split_trace_ls = split_trace(trace, split_points)

        h, x = compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                    incidence_matrix, ini,
                                    fin_vec,
                                    lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP if int_sol else lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                    split_trace_ls, use_cvxopt=True, heuristic="EXTENDED_STATE_EQUATION",
                                    int_sol=True, k=len(split_points) + 1)

        ini_state = utils.SearchTupleExtStateEq(0 + h, 0, h, ini, None, None, x, True, 0)
        # init priority queue Q, sorted ascending by f = g + h
        open_set = [ini_state]
        open_set_custom_queue = utils.MappedQueue(open_set)

        visited = 0
        queued = 0
        traversed = 0
        lp_solved += 1

        num_explained_events = 0

        trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

        while not len(open_set_custom_queue) == 0:
            #if (time.time() - start_time) > max_align_time_trace:
             #   return None

            curr = open_set_custom_queue.pop()

            current_marking = curr.m

            # current marking can be equal to the final marking only if the heuristics
            # (underestimation of the remaining cost) is 0. Low-hanging fruits
            if curr.h < 0.01:
                # if head represents final marking: return corresp. alignment by reconstructing
                if current_marking == fin:
                    return utils.__reconstruct_alignment(curr, visited, queued, traversed,
                                                         ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                         lp_solved=lp_solved)
            else:
                assert not current_marking == fin


            # 12/10/2019: do it again, since the marking could be changed
            already_closed = current_marking in closed
            if already_closed:
                continue
                
            if not curr.trust:
                # restart search with longer list of split points
                if num_explained_events not in split_points and num_explained_events > 0 and num_explained_events < len(trace):
                    split_points.append(num_explained_events)
                    split_points = sorted(split_points)
                    break

                # compute true estimate
                curr_explained = curr.num_explained
                subset_splitpoints = sorted([split_idx for split_idx in split_points if split_idx >= curr_explained])
                subset_split_trace = split_trace(trace, subset_splitpoints)[1:] #remove sigma_0

                if len(subset_split_trace) != 0:
                    h, x = compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                            incidence_matrix, curr.m,
                                            fin_vec,
                                            lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP if int_sol else lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                            subset_split_trace,
                                            use_cvxopt=use_cvxopt,
                                            heuristic = "EXTENDED_STATE_EQUATION",
                                            k = len(subset_split_trace))
                else:
                    h, x = compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                                incidence_matrix, curr.m,
                                                fin_vec,
                                                lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP if int_sol else lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                                use_cvxopt=use_cvxopt)
                
                lp_solved += 1

                tp = utils.SearchTupleExtStateEq(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True, curr.num_explained)
                added_back = open_set_custom_queue.push(tp)
                assert added_back
                continue

            # max allowed heuristics value (27/10/2019, due to the numerical instability of some of our solvers)
            if curr.h > lp_solver.MAX_ALLOWED_HEURISTICS:
                continue

            closed.add(current_marking)
            visited += 1

            num_explained_events = max(num_explained_events, curr.num_explained) # how do we determine explained events?

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

                # cost to get to the new marking
                g = curr.g + cost

                # heuristic with the new path
                h, x = utils.__derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
                trustable = utils.__trust_solution(x)
                
                # re-use exact heuristics
                if not trustable and new_marking in open_set_custom_queue.d and open_set_custom_queue.h[open_set_custom_queue.d[new_marking]].trust:
                    h = open_set_custom_queue.h[open_set_custom_queue.d[new_marking]].h
                    x = open_set_custom_queue.h[open_set_custom_queue.d[new_marking]].x
                    trustable = True
                
                new_f = g + h

                # print("Explored new marking - curr f =", curr.f, (curr.g, curr.h), "next f =", new_f, (g, h), "trusted?", trustable)

                # update num explained
                new_num_explained = curr.num_explained
                if curr.num_explained < len(trace):
                    next_label_to_explain = trace[curr.num_explained]
                    if t.label[0] == next_label_to_explain:              # what is trace net, wat is model net?
                        new_num_explained += 1

                tp = utils.SearchTupleExtStateEq(new_f, g, h, new_marking, curr, t, x, trustable, new_num_explained)
  
                # if in Q, replace if lower path cost
                # if nothing exists in Q, insert with path cost and heuristic value
                if str(new_marking) not in open_set_custom_queue.d:
                    added_back = open_set_custom_queue.push(tp)
                    queued += 1
                elif g < open_set_custom_queue.h[open_set_custom_queue.d[str(new_marking)]].g:
                    open_set_custom_queue.update(str(new_marking), tp)


def search_extended_marking_eq_faster(sync_net, ini, fin, cost_function, skip, trace, activity_key, trace_net,
                                      ret_tuple_as_trans_desc=False, max_align_time_trace=sys.maxsize, int_sol=False,
                                      solver=None):
    start_time = timer()
    lp_time = 0.0
    verbose = False

    if verbose:
        print("Starting Ext State Eq A* search with trace len", len(trace))

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

    # init k for k-based underestimation
    split_points = []
    lp_solved = 0
    solved_extended_state_eqs = 0
    round_1 = True

    while True:  # no solution
        # init set C "closed" which contains already visited markings
        closed = set()

        # store trusted heuristics to avoid re-computation
        trusted_h = {}

        split_trace_ls = split_trace(trace, split_points)

        _lp_timer_start = timer()
        h, x = compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                       incidence_matrix, ini,
                                       fin_vec,
                                       lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP if int_sol else lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                       split_trace_ls, use_cvxopt=True, heuristic="EXTENDED_STATE_EQUATION",
                                       int_sol=True, k=len(split_points) + 1)
        lp_time += timer() - _lp_timer_start

        if not round_1:
            if verbose:
                print("solution vector difference:", np.sum(np.abs(np.array(x) - np.array(old_start_x))))
        else:
            round_1 = False
        old_start_x = x

        if verbose:
            print("Solution for current ext state eq is:", (h, np.sum(np.array(x))))

        ini_state = utils.SearchTupleExtStateEq(0 + h, 0, h, ini, None, None, x, True, 0)
        # init priority queue Q, sorted ascending by f = g + h
        open_set = [ini_state]
        trusted_h[ini] = h, x
        heapq.heapify(open_set)
        visited = 0
        queued = 0
        traversed = 0
        lp_solved += 1
        solved_extended_state_eqs += 1

        num_explained_events = 0

        trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)

        while not len(open_set) == 0:
            # if (time.time() - start_time) > max_align_time_trace:
            #   return None

            curr = heapq.heappop(open_set)

            current_marking = curr.m

            # current marking can be equal to the final marking only if the heuristics
            # (underestimation of the remaining cost) is 0. Low-hanging fruits
            if curr.h < 0.01:
                # if head represents final marking: return corresp. alignment by reconstructing
                if current_marking == fin:
                    # print("lp solving runtime percentage:", lp_time / (timer() - start_time))
                    return utils.__reconstruct_alignment(curr, visited, queued, traversed,
                                                         ret_tuple_as_trans_desc=ret_tuple_as_trans_desc,
                                                         lp_solved=lp_solved)

            # 12/10/2019: do it again, since the marking could be changed
            already_closed = current_marking in closed
            if already_closed:
                continue

            if not curr.trust:
                # check if we actually do trust now
                actually_do_trust = curr.m in trusted_h

                # restart search with longer list of split points
                if not actually_do_trust and num_explained_events not in split_points and num_explained_events > 0 and num_explained_events < len(
                        trace):
                    split_points.append(num_explained_events)
                    split_points = sorted(split_points)

                    if verbose:
                        print("Restarting search with new splitpoints:", split_points)
                        print("Solved LPs so far:", lp_solved, "extended:", solved_extended_state_eqs)

                    break

                if actually_do_trust:
                    h, x = trusted_h[curr.m]
                else:
                    # compute true estimate
                    curr_explained = curr.num_explained
                    subset_splitpoints = sorted(
                        [split_idx for split_idx in split_points if split_idx >= curr_explained])
                    subset_split_trace = split_trace(trace, subset_splitpoints)[1:]  # remove sigma_0

                    _lp_timer_start = timer()
                    if False and len(subset_split_trace) != 0:
                        h, x = compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                                       incidence_matrix, curr.m,
                                                       fin_vec,
                                                       lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP if int_sol else lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                                       subset_split_trace,
                                                       use_cvxopt=use_cvxopt,
                                                       heuristic="EXTENDED_STATE_EQUATION",
                                                       k=len(subset_split_trace))
                        solved_extended_state_eqs += 1
                    else:
                        h, x = compute_exact_heuristic(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
                                                       incidence_matrix, curr.m,
                                                       fin_vec,
                                                       lp_solver.CVXOPT_SOLVER_CUSTOM_ALIGN_ILP if int_sol else lp_solver.DEFAULT_LP_SOLVER_VARIANT,
                                                       use_cvxopt=use_cvxopt)
                    lp_time += timer() - _lp_timer_start
                    lp_solved += 1

                    # this is the first time we trust you <3
                    trusted_h[curr.m] = h, x

                tp = utils.SearchTupleExtStateEq(curr.g + h, curr.g, h, curr.m, curr.p, curr.t, x, True,
                                                 curr.num_explained)
                heapq.heappush(open_set, tp)
                continue

            # max allowed heuristics value (27/10/2019, due to the numerical instability of some of our solvers)
            if curr.h > lp_solver.MAX_ALLOWED_HEURISTICS:
                continue

            closed.add(current_marking)
            visited += 1

            num_explained_events = max(num_explained_events, curr.num_explained)

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

                # update num explained
                new_num_explained = curr.num_explained
                if curr.num_explained < len(trace):
                    next_label_to_explain = trace[curr.num_explained]
                    if t.label[0] == next_label_to_explain:  # what is trace net, wat is model net?
                        new_num_explained += 1

                # if in Q, replace if lower path cost
                # if nothing exists in Q, insert with path cost and heuristic value
                g = curr.g + cost

                queued += 1
                h, x = utils.__derive_heuristic(incidence_matrix, cost_vec, curr.x, t, curr.h)
                trustable = utils.__trust_solution(x)
                new_f = g + h

                # try to get prev computed trusted heuristic
                known_trusted = new_marking in trusted_h

                if not trustable:
                    if known_trusted:
                        trustable = True
                        h, x = trusted_h[new_marking]
                elif not known_trusted:
                    trusted_h[new_marking] = h, x

                tp = utils.SearchTupleExtStateEq(new_f, g, h, new_marking, curr, t, x, trustable, new_num_explained)
                heapq.heappush(open_set, tp)

def split_trace(trace, splitpoints):

    if not splitpoints:
        return [trace]

    splitted_trace = []

    if len(splitpoints) == 1:
        splitted_trace.append(trace[:splitpoints[0]])
        splitted_trace.append(trace[splitpoints[0]:])
        return splitted_trace

    for i in range(len(splitpoints)):
        if i == 0:
            splitted_trace.append(trace[:splitpoints[i]])
        elif i == (len(splitpoints) - 1):
            splitted_trace.append(trace[splitpoints[i - 1]:splitpoints[i]])
            splitted_trace.append(trace[splitpoints[i]:])
        else:
            splitted_trace.append(trace[splitpoints[i-1]:splitpoints[i]])

    return splitted_trace


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
        # obj, x = utils.__compute_exact_extended_state_equation_ilp(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
        #                                                          incidence_matrix, marking, fin_vec, variant,
        #                                                          trace_division, ilp=int_sol,
        #                                                          use_cvxopt=use_cvxopt, k=k)

        # obj_1, x_1 = utils.__compute_exact_extended_state_equation_sparse_setup(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
        #                                                          incidence_matrix, marking, fin_vec, variant,
        #                                                          trace_division, ilp=int_sol,
        #                                                          use_cvxopt=use_cvxopt, k=k)
        
        # obj_2, x_2 = utils.__compute_exact_extended_state_equation_antons_setup(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
        #                                                          incidence_matrix, marking, fin_vec, variant,
        #                                                          trace_division, ilp=int_sol,
        #                                                          use_cvxopt=use_cvxopt, k=k)
        
        # print(obj, obj_1, obj_2)
        # print(x, x_1, x_2)

        return utils.__compute_exact_extended_state_equation_sparse_setup(sync_net, a_matrix, h_cvx, g_matrix, cost_vec,
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
