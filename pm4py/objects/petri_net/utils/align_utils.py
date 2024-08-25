'''
    This file is part of PM4Py (More Info: https://pm4py.fit.fraunhofer.de).

    PM4Py is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    PM4Py is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with PM4Py.  If not, see <https://www.gnu.org/licenses/>.
'''
import heapq
import sys
from timeit import default_timer as timer
from copy import copy
from typing import List, Tuple

import math
import numpy as np
from scipy.sparse import coo_array, csc_array
import scipy
import scipy.sparse

from pm4py.objects.petri_net import semantics, properties
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.util.lp import solver as lp_solver

SKIP = '>>'
STD_MODEL_LOG_MOVE_COST = 10000
STD_TAU_COST = 1
STD_SYNC_COST = 0

this_options_lp = {}
this_options_lp["LPX_K_MSGLEV"] = 0
this_options_lp["msg_lev"] = "GLP_MSG_OFF"
this_options_lp["show_progress"] = False
this_options_lp["presolve"] = "GLP_ON"
this_options_lp["tm_lim"] = 60000
this_options_lp["pp_tech"] = "GLP_PP_NONE"


def search_path_among_sol(sync_net: PetriNet, ini: Marking, fin: Marking,
                          activated_transitions: List[PetriNet.Transition], skip=SKIP) -> Tuple[
    List[PetriNet.Transition], bool, int]:
    """
    (Efficient method) Searches a firing sequence among the X vector that is the solution of the
    (extended) marking equation

    Parameters
    ---------------
    sync_net
        Synchronous product net
    ini
        Initial marking of the net
    fin
        Final marking of the net
    activated_transitions
        Transitions that have non-zero occurrences in the X vector
    skip
        Skip transition

    Returns
    ---------------
    firing_sequence
        Firing sequence
    reach_fm
        Boolean value that tells if the final marking is reached by the firing sequence
    explained_events
        Number of explained events
    """
    reach_fm = False
    trans_empty_preset = set(t for t in sync_net.transitions if len(t.in_arcs) == 0)
    trans_with_index = {}
    trans_wo_index = set()
    for t in activated_transitions:
        if properties.TRACE_NET_TRANS_INDEX in t.properties:
            trans_with_index[t.properties[properties.TRACE_NET_TRANS_INDEX]] = t
        else:
            trans_wo_index.add(t)
    keys = sorted(list(trans_with_index.keys()))
    trans_with_index = [trans_with_index[i] for i in keys]
    best_tuple = (0, 0, ini, list())
    open_set = [best_tuple]
    heapq.heapify(open_set)
    visited = 0
    closed = set()
    len_trace_with_index = len(trans_with_index)
    while len(open_set) > 0:
        curr = heapq.heappop(open_set)
        index = -curr[0]
        marking = curr[2]
        if marking in closed:
            continue
        if index == len_trace_with_index:
            reach_fm = True
            if curr[0] < best_tuple[0]:
                best_tuple = curr
            break
        if curr[0] < best_tuple[0]:
            best_tuple = curr
        closed.add(marking)
        corr_trans = trans_with_index[index]
        if corr_trans.sub_marking <= marking:
            visited += 1
            new_marking = semantics.weak_execute(corr_trans, marking)
            heapq.heappush(open_set, (-index - 1, visited, new_marking, curr[3] + [corr_trans]))
        else:
            enabled = copy(trans_empty_preset)
            for p in marking:
                for t in p.ass_trans:
                    if t in trans_wo_index and t.sub_marking <= marking:
                        enabled.add(t)
            for new_trans in enabled:
                visited += 1
                new_marking = semantics.weak_execute(new_trans, marking)
                heapq.heappush(open_set, (-index, visited, new_marking, curr[3] + [new_trans]))
    return best_tuple[-1], reach_fm, -best_tuple[0]


def construct_standard_cost_function(synchronous_product_net, skip):
    """
    Returns the standard cost function, which is:
    * event moves: cost 1000
    * model moves: cost 1000
    * tau moves: cost 1
    * sync moves: cost 0
    :param synchronous_product_net:
    :param skip:
    :return:
    """
    costs = {}
    for t in synchronous_product_net.transitions:
        if (skip == t.label[0] or skip == t.label[1]) and (t.label[0] is not None and t.label[1] is not None):
            costs[t] = STD_MODEL_LOG_MOVE_COST
        else:
            if skip == t.label[0] and t.label[1] is None:
                costs[t] = STD_TAU_COST
            else:
                costs[t] = STD_SYNC_COST
    return costs


def pretty_print_alignments(alignments):
    """
    Takes an alignment and prints it to the console, e.g.:
     A  | B  | C  | D  |
    --------------------
     A  | B  | C  | >> |
    :param alignment: <class 'list'>
    :return: Nothing
    """
    if isinstance(alignments, list):
        for alignment in alignments:
            __print_single_alignment(alignment["alignment"])
    else:
        __print_single_alignment(alignments["alignment"])


def __print_single_alignment(step_list):
    trace_steps = []
    model_steps = []
    max_label_length = 0
    for step in step_list:
        trace_steps.append(" " + str(step[0]) + " ")
        model_steps.append(" " + str(step[1]) + " ")
        if len(step[0]) > max_label_length:
            max_label_length = len(str(step[0]))
        if len(str(step[1])) > max_label_length:
            max_label_length = len(str(step[1]))
    for i in range(len(trace_steps)):
        if len(str(trace_steps[i])) - 2 < max_label_length:
            step_length = len(str(trace_steps[i])) - 2
            spaces_to_add = max_label_length - step_length
            for j in range(spaces_to_add):
                if j % 2 == 0:
                    trace_steps[i] = trace_steps[i] + " "
                else:
                    trace_steps[i] = " " + trace_steps[i]
        print(trace_steps[i], end='|')
    divider = ""
    length_divider = len(trace_steps) * (max_label_length + 3)
    for i in range(length_divider):
        divider += "-"
    print('\n' + divider)
    for i in range(len(model_steps)):
        if len(model_steps[i]) - 2 < max_label_length:
            step_length = len(model_steps[i]) - 2
            spaces_to_add = max_label_length - step_length
            for j in range(spaces_to_add):
                if j % 2 == 0:
                    model_steps[i] = model_steps[i] + " "
                else:
                    model_steps[i] = " " + model_steps[i]

        print(model_steps[i], end='|')
    print('\n\n')


def add_markings(curr, add):
    m = Marking()
    for p in curr.items():
        m[p[0]] = p[1]
    for p in add.items():
        m[p[0]] += p[1]
        if m[p[0]] == 0:
            del m[p[0]]
    return m


def __get_alt(open_set, new_marking):
    for item in open_set:
        if item.m == new_marking:
            return item


def __reconstruct_alignment(state, visited, queued, traversed, ret_tuple_as_trans_desc=False, lp_solved=0):
    alignment = list()
    if state.p is not None and state.t is not None:
        parent = state.p
        if ret_tuple_as_trans_desc:
            alignment = [(state.t.name, state.t.label)]
            while parent.p is not None:
                alignment = [(parent.t.name, parent.t.label)] + alignment
                parent = parent.p
        else:
            alignment = [state.t.label]
            while parent.p is not None:
                alignment = [parent.t.label] + alignment
                parent = parent.p
    return {'alignment': alignment, 'cost': state.g, 'visited_states': visited, 'queued_states': queued,
            'traversed_arcs': traversed, 'lp_solved': lp_solved}


def __derive_heuristic(incidence_matrix, cost_vec, x, t, h):
    x_prime = x.copy()
    x_prime[incidence_matrix.transitions[t]] -= 1
    return max(0, h - cost_vec[incidence_matrix.transitions[t]]), x_prime


def __is_model_move(t, skip):
    return t.label[0] == skip and t.label[1] != skip


def __is_log_move(t, skip):
    return t.label[0] != skip and t.label[1] == skip


def __trust_solution(x):
    for v in x:
        if v < -0.001:
            return False
    return True


def __compute_exact_heuristic_new_version(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                          marking, fin_vec, variant, use_cvxopt=False, strict=True):
    
    state_eq_setup_time_start = timer()

    m_vec = incidence_matrix.encode_marking(marking)
    b_term = [i - j for i, j in zip(fin_vec, m_vec)]
    b_term = np.matrix([x * 1.0 for x in b_term]).transpose()

    if not strict:
        g_matrix = np.vstack([g_matrix, a_matrix])
        h_cvx = np.vstack([h_cvx, b_term])
        a_matrix = np.zeros((0, a_matrix.shape[1]))
        b_term = np.zeros((0, b_term.shape[1]))

    if use_cvxopt:
        # not available in the latest version of PM4Py
        from cvxopt import matrix

        b_term = matrix(b_term)

    parameters_solving = {"solver": "glpk"}

    state_eq_setup_time = timer() - state_eq_setup_time_start

    # print("STATE EQ. SETUP TIME:", state_eq_setup_time)
    state_eq_solving_time_start = timer()

    sol = lp_solver.apply(cost_vec, g_matrix, h_cvx, a_matrix, b_term, parameters=parameters_solving,
                          variant=variant)
    prim_obj = lp_solver.get_prim_obj_from_sol(sol, variant=variant)
    points = lp_solver.get_points_from_sol(sol, variant=variant)

    prim_obj = prim_obj if prim_obj is not None else sys.maxsize
    points = points if points is not None else [0.0] * len(sync_net.transitions)

    # print("STATE EQ. SOLVING TIME:", timer() - state_eq_solving_time_start)
    return prim_obj, points


def __compute_exact_extended_state_equation(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                            marking, fin_vec, variant, trace_division, use_cvxopt=False, strict=True,
                                            k=1):
    extended_state_eq_setup_time_start = timer()

    # lp version

    # trace_divison = [substring_1, ..., substring_k]

    # return solution vector z = x_0 + sum ( x_a + y_a )
    # return estimate h
    from cvxopt.modeling import op, variable, dot, sum, matrix

    len_transitions = len(incidence_matrix.transitions)
    len_places = len(incidence_matrix.places)

    # variable y_i refers to firing of single transition at start of each subtrace
    # in y_i starting with transition in trace model
    # y_0 possibly empty prefix of transitions of form (t_m, skip)

    # y = np.empty((k, len_transitions))

    y = [0 for i in range(k + 1)]

    for i in range(1, k + 1):
        y[i] = variable(size=len_transitions, name="y_%s" % i)

    # variable x_i corresponds to any other transitions firing

    # x = np.empty((k+1, len_transitions))

    x = [0 for i in range(k + 1)]

    for i in range(k + 1):
        x[i] = variable(size=len_transitions, name="x_%s" % i)

    # objective function
    cost_transp = np.transpose(cost_vec)

    sum_x = x[0]
    for i in range(1, k + 1):
        sum_x = sum_x + x[i]

    sum_y = y[1]
    for i in range(2, k + 1):
        sum_y = sum_y + y[i]

    # obj = np.dot(cost_transp, np.sum(x.value, axis=0)) + np.dot(cost_transp, np.sum(y.value, axis=0))
    # dot transposes first input
    obj = dot(cost_vec, sum_x) + dot(cost_vec, sum_y)

    model = op(obj)

    # c1: marking eq., reach final marking from initial/current after combining all firing transitions of all x_a and y_a
    # was, wenn mittendrin berechnet wird? Trotzdem von initial marking?
    m_vec = incidence_matrix.encode_marking(marking)
    m_vec_2d = [m_vec[i:i + 1] for i in range(0, len(m_vec), 1)]
    m_matrix = matrix(m_vec_2d, (len_places, 1), tc='d')

    fin_vec_2d = [fin_vec[i:i + 1] for i in range(0, len(fin_vec), 1)]
    fin_matrix = matrix(fin_vec_2d, (len_places, 1), tc='d')

    sums_transitions = x[1] + y[1]

    for i in range(2, k + 1):
        sums_transitions = sums_transitions + x[i] + y[i]
    # op.addconstraint(m_matrix + matrix(a_matrix) * x[0] + matrix(a_matrix) * sums_transitions == fin_matrix)

    m_a_matrix = matrix(a_matrix, tc='d')

    constr1 = (m_matrix + m_a_matrix * x[0] + m_a_matrix * sums_transitions == fin_matrix)
    model.addconstraint(constr1)
    # c2: extended marking eq., after firing prefix of transitions, sufficient tokens available to fire first transition in y_a

    # consumption matrix is incidence matrix without positive entries
    consumption_matrix = np.copy(incidence_matrix.a_matrix)

    for i in range(consumption_matrix.shape[0]):
        for j in range(consumption_matrix.shape[1]):
            if consumption_matrix[i][j] > 0:
                consumption_matrix[i][j] = 0

    consumption_matrix = matrix(consumption_matrix, tc='d')

    for a in range(1, k + 1):
        sum_transitions_subsequences = np.zeros((len_transitions, 1))
        # sum_transitions_subsequences = [0] * len_transitions
        for b in range(1, a):
            if b == 1:
                sum_transitions_subsequences_2 = x[b] + y[b]
            else:
                sum_transitions_subsequences_2 = sum_transitions_subsequences_2 + x[b] + y[b]

        if a < 2:
            m_sum_transitions_subsequences = matrix(sum_transitions_subsequences, tc='d')
            model.addconstraint(matrix(np.zeros_like(m_vec), tc='d')
                                <= m_matrix + m_a_matrix * x[0] + m_a_matrix * m_sum_transitions_subsequences +
                                consumption_matrix * y[a])
        else:
            # m_sum_transitions_subsequences_2 = matrix(sum_transitions_subsequences_2, tc='d')
            model.addconstraint(matrix(np.zeros_like(m_vec), tc='d')
                                <= m_matrix + m_a_matrix * x[0] + m_a_matrix * sum_transitions_subsequences_2 +
                                consumption_matrix * y[a])

    # c3: x_a is natural number, relax to real value numbers
    for a in range(k + 1):
        model.addconstraint(0 <= x[a])

    # c4: every element of y_a is 0 or 1, relax to real value numbers
    for a in range(1, k + 1):
        model.addconstraint(0 <= y[a])
        model.addconstraint(y[a] <= 1)

    # c6: only one element of y_a equals 1
    for a in range(1, k + 1):
        model.addconstraint(dot(matrix(np.ones((len_transitions, 1)), tc='d'), y[a]) == 1)
        # model.addconstraint(np.dot(np.ones((1, len_transitions)), y[a]) == 1)

    # c5: y_a corresponds to transition of synchronous product corresp. to start of subtrace_a
    # all transitions not corresponding to start of subtrace_a in y_a are 0
    # how to get i-th transition of log moves?
    # ((t_trace.name, t_model.name), (t_trace.label, t_model.label))

    transitions = []
    for t in incidence_matrix.transitions:
        transitions.append(t)

    for a in range(1, k + 1):
        for t in range(len(transitions)):
            if transitions[t].label[0] != trace_division[a - 1][0]:
                model.addconstraint(y[a][t] == 0)

    extended_state_eq_setup_time = timer() - extended_state_eq_setup_time_start
    print("EXTENDED STATE EQ SETUP TIME:", extended_state_eq_setup_time)
    extended_state_eq_solve_time_start = timer()
    model.solve(solver='glpk')

    # prim_obj corresponds to underestimate h
    prim_obj = model.objective.value()

    prim_obj = prim_obj[0] if prim_obj is not None else sys.maxsize
    # points = points if points is not None else [0.0] * len(sync_net.transitions)

    # points corresponds to solution vector z = x_0 + sum over (x_a + y_a) for 1<= a <= k

    if not model.status == "primal infeasible":
        points = x[0]
        for a in range(1, k + 1):
            sum_subsequence = x[a] + y[a]
            points = points + sum_subsequence

        # solution vector as list of values
        points_list = []
        for i in range(len_transitions):
            points_list.append(points.value()[i])
    else:
        points_list = [0.0] * len_transitions

    extended_state_eq_solve_time = timer() - extended_state_eq_solve_time_start
    print("EXTENDED STATE EQ SOLVE TIME:", extended_state_eq_solve_time)
    return prim_obj, points_list


def check_lp_sol_int(x):
    for i in range(len(x)):
        if abs(x[i] - round(x[i])) > 10 ** (-5):
            return False
    return True


def compute_extended_state_equation_res(res_vec, y_indices, num_y_variables, len_transitions, k):
    x_part = np.array(res_vec[:-num_y_variables])
    y_part = np.array(res_vec[-num_y_variables:])
    res = np.sum(np.reshape(x_part, newshape=(k + 1, len_transitions)), axis=0)
    
    running_i = 0
    for y_index in y_indices:
        curr_cost_vec = np.zeros(len_transitions)
        for y_id in y_index:
            curr_cost_vec[y_id] = y_part[running_i]
            running_i += 1
        res += curr_cost_vec
    return res

def __compute_exact_extended_state_equation_a_setup(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                                         marking, fin_vec, variant, trace_division, ilp=False, use_cvxopt=False,
                                                         strict=True,
                                                         k=1):
    
    assert k == len(trace_division)
    setup_time_start = timer()
    len_transitions = len(incidence_matrix.transitions)
    len_places = len(incidence_matrix.places)

    ##################################
    #       Y variable indices       #
    ##################################

    y_indices = [np.array([transition_idx for transition_idx, transition in enumerate(incidence_matrix.transitions) if transition.label[0] == trace_division[a][0]]) for a in range(k)]
    y_indices_prefix_sum = [[]] + [np.concatenate(list(y_indices[:i+1])) for i in range(len(y_indices) - 1)]
    y_idx_len_prefix_sum = [len(y_pre) for y_pre in y_indices_prefix_sum]
    num_y_variables = sum([len(y_ids) for y_ids in y_indices])

    ##################################
    #          COST VECTOR           #
    ##################################

    c_vec = np.asarray(cost_vec).flatten()
    x_costs = np.tile(c_vec, k + 1)
    y_costs = np.concatenate([c_vec[_y_id] for _y_id in y_indices])
    ext_cost_vec = np.concatenate([x_costs, y_costs])

    ##################################
    #      EQUALITY CONSTRAINTS      #
    ##################################

    #transition matrix
    np_C = np.asarray(a_matrix, dtype=np.float64)
    C_x = np.tile(np_C, k + 1)
    C_y = np.concatenate([np_C[:, y_id] for y_id in y_indices], axis=1)
    ext_C = np.concatenate([C_x, C_y], axis=1)

    # y-constraints (must sum to one)
    y_constraints = np.vstack([np.concatenate([np.zeros((k + 1) * len_transitions + pref_len),
                                               np.ones(len(y_idx)),
                                               np.zeros(num_y_variables - pref_len - len(y_idx))]) 
                               for pref_len, y_idx in zip(y_idx_len_prefix_sum, y_indices)])

    ext_C = np.vstack([ext_C, y_constraints])

    # eq rhs
    m_vec = incidence_matrix.encode_marking(marking)
    b_term = np.asarray([1.0 * (i - j) for i, j in zip(fin_vec, m_vec)])
    eq_rhs = np.concatenate([b_term, np.ones(k)])

    ##################################
    #     INEQUALITY CONSTRAINTS     #
    ##################################

    neg_transition_mat = -np_C
    neg_cost_mat = -np.asarray(incidence_matrix.consumption_matrix, dtype=np.float64)
    x_consumption = np.vstack([np.concatenate([np.tile(neg_transition_mat, 1 + a), np.zeros(shape=(len_places, (k-a) * len_transitions))], axis=1) for a in range(k)])
    y_consumption = np.vstack([np.concatenate([neg_transition_mat[:, pref], 
                                              neg_cost_mat[:,y_idx], 
                                              np.zeros(shape=(len_places, num_y_variables - pref_len - len(y_idx)))], axis=1) 
                                              for pref, pref_len, y_idx in zip(y_indices_prefix_sum, y_idx_len_prefix_sum, y_indices)])
    consumption_cstr = np.concatenate([x_consumption, y_consumption], axis=1)
    x_nonneg_constr = np.concatenate([-np.eye((k+1) * len_transitions), np.zeros(shape=((k+1) * len_transitions, num_y_variables))], axis=1)
    y_nonneg_constr = np.concatenate([np.zeros(shape=(num_y_variables, (k+1) * len_transitions)), -np.eye(num_y_variables)], axis=1)
    y_max_one_constr = -y_nonneg_constr
    ineq_mat = np.vstack([consumption_cstr, x_nonneg_constr, y_nonneg_constr, y_max_one_constr])
    ineq_rhs = np.concatenate([np.tile(m_vec, k), np.zeros((k + 1) * len_transitions + num_y_variables), np.ones(num_y_variables)])

    setup_time = timer() - setup_time_start
    # print("TIME FOR MATRIX SETUP:", setup_time)

    ##################################
    #              SOLVE             #
    ##################################

    solving_time_start = timer()

    from cvxopt import matrix, sparse

    parameters_solving = {"solver": "glpk"}
    sol = lp_solver.apply(matrix(ext_cost_vec), matrix(ineq_mat), matrix(np.expand_dims(ineq_rhs, axis=-1)), matrix(ext_C), matrix(np.expand_dims(eq_rhs, axis=-1)), parameters=parameters_solving,
                          variant=variant)
    prim_obj = lp_solver.get_prim_obj_from_sol(sol, variant=variant)
    points = lp_solver.get_points_from_sol(sol, variant=variant)
    
    solution_found = prim_obj is not None
    prim_obj = prim_obj if prim_obj is not None else sys.maxsize
    points = compute_extended_state_equation_res(points, y_indices, num_y_variables, len_transitions, k) if points is not None else [0.0] * len(sync_net.transitions)

    if not solution_found:
        print("WARNING! NO SOLUTION TO EXTENDED STATE EQ FOUND!")

    # print("MATRIX EXT STATE EQ SOLVE TIME:", timer() - solving_time_start)

    # print("Equality Matriy density:", np.count_nonzero(ext_C) / np.prod(ext_C.shape), "Size:", np.prod(ext_C.shape))
    # print("Inequality Matriy density:", np.count_nonzero(ineq_mat) / np.prod(ineq_mat.shape), np.prod(ineq_mat.shape))

    # print("num transitions:", len_transitions)

    # print("eq_rhs shape:", eq_rhs.shape)
    # print("eq mat shape:", ext_C.shape)

    # print("ineq_rhs shape:", ineq_rhs.shape)
    # print("ineq mat shape:", ineq_mat.shape)

    # print("Primal objective:", prim_obj)
    # print("Points:", points)
    # print("Points shape:", points.shape)

    return prim_obj, points


def scipy_sparse_to_cvxopt_sparse(mat):
    from cvxopt import spmatrix
    mat = scipy.sparse.coo_matrix(mat)
    return spmatrix(mat.data, mat.row, mat.col, mat.shape)

def __compute_exact_extended_state_equation_sparse_setup(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                                         marking, fin_vec, variant, trace_division, ilp=False, use_cvxopt=False,
                                                         strict=True,
                                                         k=1):
    
    assert k == len(trace_division)
    setup_time_start = timer()
    len_transitions = len(incidence_matrix.transitions)
    len_places = len(incidence_matrix.places)

    ##################################
    #       Y variable indices       #
    ##################################

    y_indices = [np.array([transition_idx for transition_idx, transition in enumerate(incidence_matrix.transitions) if transition.label[0] == trace_division[a][0]]) for a in range(k)]
    y_indices_prefix_sum = [[]] + [np.concatenate(list(y_indices[:i+1])) for i in range(len(y_indices) - 1)]
    y_idx_len_prefix_sum = [len(y_pre) for y_pre in y_indices_prefix_sum]
    num_y_variables = sum([len(y_ids) for y_ids in y_indices])

    ##################################
    #          COST VECTOR           #
    ##################################

    c_vec = np.asarray(cost_vec).flatten()
    x_costs = np.tile(c_vec, k + 1)
    y_costs = np.concatenate([c_vec[_y_id] for _y_id in y_indices])
    ext_cost_vec = np.concatenate([x_costs, y_costs])

    ##################################
    #      EQUALITY CONSTRAINTS      #
    ##################################

    #transition matrix
    np_C = np.asarray(a_matrix)
    sp_C_coo = scipy.sparse.coo_array(np_C)
    sp_C_csc = scipy.sparse.csc_array(np_C)
    C_x = scipy.sparse.hstack([sp_C_coo] * (k + 1))
    C_y = scipy.sparse.hstack([sp_C_csc[:, y_id] for y_id in y_indices])
    ext_C = scipy.sparse.hstack([C_x, C_y])

    # y-constraints (must sum to one)

    y_constraints = scipy.sparse.vstack(
        [scipy.sparse.coo_matrix(([1] * len(y_idx), 
                                  ([0] * len(y_idx), 
                                   [i for i in range((k + 1) * len_transitions + pref_len, (k + 1) * len_transitions + pref_len + len(y_idx))])),
                                   shape=(1,(k+1) * len_transitions + num_y_variables)) 
        for pref_len, y_idx in zip(y_idx_len_prefix_sum, y_indices)]
    )

    # y_constraints_dense = np.vstack([np.concatenate([np.zeros((k + 1) * len_transitions + pref_len),
    #                                            np.ones(len(y_idx)),
    #                                            np.zeros(num_y_variables - pref_len - len(y_idx))]) 
    #                            for pref_len, y_idx in zip(y_idx_len_prefix_sum, y_indices)])

    # assert (y_constraints.toarray() == y_constraints_dense).all()

    ext_C = scipy.sparse.vstack([ext_C, y_constraints])

    # eq rhs
    m_vec = incidence_matrix.encode_marking(marking)
    b_term = np.asarray([1.0 * (i - j) for i, j in zip(fin_vec, m_vec)])
    eq_rhs = np.concatenate([b_term, np.ones(k)])

    ##################################
    #     INEQUALITY CONSTRAINTS     #
    ##################################

    neg_transition_mat = -np_C
    neg_cost_mat = -np.asarray(incidence_matrix.consumption_matrix, dtype=np.float64)
    neg_transition_mat_sparse_csc = scipy.sparse.csc_matrix(neg_transition_mat)
    neg_cost_mat_sparse_csc = scipy.sparse.csc_matrix(neg_cost_mat)
    x_consumption = scipy.sparse.vstack([scipy.sparse.hstack((1 + a) * [neg_transition_mat_sparse_csc] + [scipy.sparse.csc_matrix((len_places, (k-a) * len_transitions))]) for a in range(k)])
    y_consumption = scipy.sparse.vstack([scipy.sparse.hstack([
            neg_transition_mat_sparse_csc[:, pref],
            neg_cost_mat_sparse_csc[:, y_idx],
            scipy.sparse.csc_matrix((len_places, num_y_variables - pref_len - len(y_idx)))
        ])
        for pref, pref_len, y_idx in zip(y_indices_prefix_sum, y_idx_len_prefix_sum, y_indices)
    ])

    consumption_cstr = scipy.sparse.hstack([x_consumption, y_consumption])
    x_nonneg_constr = scipy.sparse.hstack([-scipy.sparse.identity((k+1) * len_transitions), scipy.sparse.coo_matrix(((k+1) * len_transitions, num_y_variables))])
    y_nonneg_constr = scipy.sparse.hstack([scipy.sparse.coo_matrix((num_y_variables, (k+1) * len_transitions)), -scipy.sparse.identity(num_y_variables)])
    y_max_one_constr = -y_nonneg_constr

    ineq_mat = scipy.sparse.vstack([consumption_cstr, x_nonneg_constr, y_nonneg_constr, y_max_one_constr])
    ineq_rhs = np.concatenate([np.tile(m_vec, k), np.zeros((k + 1) * len_transitions + num_y_variables), np.ones(num_y_variables)])

    setup_time = timer() - setup_time_start

    ineq_mat_cvxopt = scipy_sparse_to_cvxopt_sparse(ineq_mat)
    eq_mat_cvxopt = scipy_sparse_to_cvxopt_sparse(ext_C)

    # print("TIME FOR MATRIX SETUP:", setup_time)

    ##################################
    #              SOLVE             #
    ##################################

    solving_time_start = timer()

    from cvxopt import matrix, sparse

    parameters_solving = {"solver": "glpk"}
    # parameters_solving = {}
    sol = lp_solver.apply(matrix(ext_cost_vec), ineq_mat_cvxopt, matrix(np.expand_dims(ineq_rhs, axis=-1)), eq_mat_cvxopt, matrix(np.expand_dims(eq_rhs, axis=-1)), parameters=parameters_solving,
                          variant=variant)
    prim_obj = lp_solver.get_prim_obj_from_sol(sol, variant=variant)
    points = lp_solver.get_points_from_sol(sol, variant=variant)
    
    # print("MATRIX EXT STATE EQ SOLVE TIME:", timer() - solving_time_start)

    extraction_timer = timer()

    solution_found = prim_obj is not None
    prim_obj = prim_obj if prim_obj is not None else sys.maxsize
    points = compute_extended_state_equation_res(points, y_indices, num_y_variables, len_transitions, k) if points is not None else [0.0] * len(sync_net.transitions)

    if not solution_found:
        print("WARNING! NO SOLUTION TO EXTENDED STATE EQ FOUND!")
        assert False

    # print("SOLUTION EXTRACTION TIME:", timer() - extraction_timer)

    # print("num transitions:", len_transitions)

    # print("eq_rhs shape:", eq_rhs.shape)
    # print("eq mat shape:", ext_C.shape)

    # print("ineq_rhs shape:", ineq_rhs.shape)
    # print("ineq mat shape:", ineq_mat.shape)

    # print("Primal objective:", prim_obj)
    # print("Points:", points)
    # print("Points shape:", points.shape)

    return prim_obj, points


def __compute_exact_extended_state_equation_matrix_setup(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                                marking, fin_vec, variant, trace_division, ilp=False, use_cvxopt=False,
                                                strict=True,
                                                k=1):
    
    assert k == len(trace_division)

    setup_time_start = timer()

    len_transitions = len(incidence_matrix.transitions)
    len_places = len(incidence_matrix.places)

    ##################################
    #          COST VECTOR           #
    ##################################

    ext_cost_vec = np.tile(np.asarray(cost_vec).flatten(), 2*k + 1)

    ##################################
    #      EQUALITY CONSTRAINTS      #
    ##################################

    #transition matrix
    np_C = np.asarray(a_matrix)

    ext_C = np.tile(np_C, 2*k + 1)

    # y-constraints
    y_start_constraints = np.concatenate([np.zeros((k+1) * len_transitions), np.array([[(0.0 if t.label[0] == trace_division[a][0] else 1.0) for t in incidence_matrix.transitions] for a in range(k)]).flatten()])
    ext_C = np.append(ext_C, np.expand_dims(y_start_constraints, axis=0), axis=0)

    ones_constraints = np.array([np.concatenate([np.zeros((k + 1 + a) * len_transitions), np.ones(len_transitions), np.zeros((k - 1 - a) * len_transitions)]) for a in range(k)])
    ext_C = np.append(ext_C, ones_constraints, axis=0)
    
    # eq rhs
    m_vec = incidence_matrix.encode_marking(marking)
    b_term = np.asarray([1.0 * (i - j) for i, j in zip(fin_vec, m_vec)])

    eq_rhs = np.concatenate([b_term, [0], np.ones(k)])

    ##################################
    #     INEQUALITY CONSTRAINTS     #
    ##################################

    neg_transition_mat = -np_C
    neg_cost_mat = -np.minimum(a_matrix, 0)
    consumption_cstr = np.vstack([np.concatenate([np.tile(neg_transition_mat, 1 + a), np.zeros(shape=(len_places, (k-a) * len_transitions)), np.tile(neg_transition_mat, a), neg_cost_mat, np.zeros(shape=(len_places, (k-a-1) * len_transitions))], axis=1) for a in range(k)])
    x_nonneg_constr = np.concatenate([-np.eye((k+1) * len_transitions), np.zeros(shape=((k+1) * len_transitions, k * len_transitions))], axis=1)
    y_nonneg_constr = np.concatenate([np.zeros(shape=(k * len_transitions, (k+1) * len_transitions)), -np.eye(k * len_transitions)], axis=1)
    y_max_one_constr = -y_nonneg_constr

    ineq_mat = np.vstack([consumption_cstr, x_nonneg_constr, y_nonneg_constr, y_max_one_constr])

    ineq_rhs = np.concatenate([np.tile(m_vec, k), np.zeros((2 * k + 1) * len_transitions), np.ones(k * len_transitions)])

    setup_time = timer() - setup_time_start
    # print("TIME FOR MATRIX SETUP:", setup_time)

    ##################################
    #              SOLVE             #
    ##################################

    solving_time_start = timer()

    from cvxopt import matrix

    parameters_solving = {"solver": "glpk"}
    sol = lp_solver.apply(matrix(ext_cost_vec), matrix(ineq_mat), matrix(np.expand_dims(ineq_rhs, axis=-1)), matrix(ext_C), matrix(np.expand_dims(eq_rhs, axis=-1)), parameters=parameters_solving,
                          variant=variant)
    prim_obj = lp_solver.get_prim_obj_from_sol(sol, variant=variant)
    points = lp_solver.get_points_from_sol(sol, variant=variant)
    
    solution_found = prim_obj is not None
    prim_obj = prim_obj if prim_obj is not None else sys.maxsize
    points = np.sum(np.reshape(points, (2*k + 1, len_transitions)), axis=0) if points is not None else [0.0] * len(sync_net.transitions)

    # print("eq_rhs shape:", eq_rhs.shape)
    # print("eq mat shape:", ext_C.shape)

    # print("ineq_rhs shape:", ineq_rhs.shape)
    # print("ineq mat shape:", ineq_mat.shape)

    # print("MATRIX EXT STATE EQ SOLVE TIME:", timer() - solving_time_start)

    # print("Primal objective:", prim_obj)
    # print("Points:", points)
    # print("Points shape:", points.shape)

    return prim_obj, points, solution_found

def __compute_exact_extended_state_equation_ilp(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                                marking, fin_vec, variant, trace_division, ilp=False, use_cvxopt=False,
                                                strict=True,
                                                k=1):
    # ilp version

    # trace_divison = [substring_1, ..., substring_k]

    # return solution vector z = x_0 + sum ( x_a + y_a )
    # return estimate h
    # st_total = time.time()

    extended_state_eq_setup_time_start = timer()

    from cvxopt.modeling import op, variable, dot, sum, matrix

    # number of solved ilps
    ilp_solved = 0
    len_transitions = len(incidence_matrix.transitions)
    len_places = len(incidence_matrix.places)

    # variable y_i refers to firing of single transition at start of each subtrace
    # in y_i starting with transition in trace model
    # y_0 possibly empty prefix of transitions of form (t_m, skip)

    # y = np.empty((k, len_transitions))

    y = [0 for i in range(k + 1)]

    for i in range(1, k + 1):
        y[i] = variable(size=len_transitions, name="y_%s" % i)

    # variable x_i corresponds to any other transitions firing

    x = [0 for i in range(k + 1)]

    for i in range(k + 1):
        x[i] = variable(size=len_transitions, name="x_%s" % i)

    # objective function
    # cost_transp = np.transpose(cost_vec)

    # st_obj = time.time()

    sum_x = x[0]
    for i in range(1, k + 1):
        sum_x = sum_x + x[i]

    sum_y = y[1]
    for i in range(2, k + 1):
        sum_y = sum_y + y[i]

    # obj = np.dot(cost_transp, np.sum(x.value, axis=0)) + np.dot(cost_transp, np.sum(y.value, axis=0))
    # dot transposes first input
    cost_vec = matrix(cost_vec)
    obj = dot(cost_vec, sum_x) + dot(cost_vec, sum_y)

    model = op(obj)

    # et_obj = time.time()
    # time_obj = et_obj - st_obj
    # c1: marking eq., reach final marking from initial/current after combining all firing transitions of all x_a and y_a
    # was, wenn mittendrin berechnet wird? Trotzdem von initial marking?

    # st_constr1 = time.time()

    m_vec = incidence_matrix.encode_marking(marking)
    # m_vec_2d = [m_vec[i:i + 1] for i in range(0, len(m_vec), 1)]
    m_vec_2d = [m_vec[i:i + 1] for i in range(len_places)]
    m_matrix = matrix(m_vec_2d, (len_places, 1), tc='d')

    # fin_vec_2d = [fin_vec[i:i + 1] for i in range(0, len(fin_vec), 1)]
    fin_vec_2d = [fin_vec[i:i + 1] for i in range(len_places)]
    fin_matrix = matrix(fin_vec_2d, (len_places, 1), tc='d')

    sums_transitions = x[1] + y[1]

    for i in range(2, k + 1):
        sums_transitions = sums_transitions + x[i] + y[i]
    # op.addconstraint(m_matrix + matrix(a_matrix) * x[0] + matrix(a_matrix) * sums_transitions == fin_matrix)

    m_a_matrix = matrix(a_matrix, tc='d')

    constr1 = (m_matrix + m_a_matrix * x[0] + m_a_matrix * sums_transitions == fin_matrix)
    model.addconstraint(constr1)

    # et_constr1 = time.time()
    # time_constr1 = et_constr1 - st_constr1

    # c2: extended marking eq., after firing prefix of transitions, sufficient tokens available to fire first transition in y_a
    # st_constr2 = time.time()

    # consumption matrix is incidence matrix without positive entries
    consumption_matrix = np.minimum(incidence_matrix.a_matrix, 0)

    consumption_matrix = matrix(consumption_matrix, tc='d')

    for a in range(1, k + 1):
        sum_transitions_subsequences = np.zeros((len_transitions, 1))
        # sum_transitions_subsequences = [0] * len_transitions
        for b in range(1, a):
            if b == 1:
                sum_transitions_subsequences_2 = x[b] + y[b]
            else:
                sum_transitions_subsequences_2 = sum_transitions_subsequences_2 + x[b] + y[b]

        if a < 2:
            m_sum_transitions_subsequences = matrix(sum_transitions_subsequences, tc='d')
            model.addconstraint(matrix(np.zeros_like(m_vec), tc='d')
                                <= m_matrix + m_a_matrix * x[0] + m_a_matrix * m_sum_transitions_subsequences +
                                consumption_matrix * y[a])
        else:
            # m_sum_transitions_subsequences_2 = matrix(sum_transitions_subsequences_2, tc='d')
            model.addconstraint(matrix(np.zeros_like(m_vec), tc='d')
                                <= m_matrix + m_a_matrix * x[0] + m_a_matrix * sum_transitions_subsequences_2 +
                                consumption_matrix * y[a])

    # et_constr2 = time.time()
    # time_constr2 = et_constr2 - st_constr2

    # c3: x_a is natural number, relax to real value numbers
    for a in range(k + 1):
        model.addconstraint(0 <= x[a])

    # c4: every element of y_a is 0 or 1, relax to real value numbers
    for a in range(1, k + 1):
        model.addconstraint(0 <= y[a])
        model.addconstraint(y[a] <= 1)

    # c6: only one element of y_a equals 1
    for a in range(1, k + 1):
        model.addconstraint(dot(matrix(np.ones((len_transitions, 1)), tc='d'), y[a]) == 1)

    # c5: y_a corresponds to transition of synchronous product corresp. to start of subtrace_a
    # all transitions not corresponding to start of subtrace_a in y_a are 0
    # how to get i-th transition of log moves?
    # ((t_trace.name, t_model.name), (t_trace.label, t_model.label))

    # st_constr5 = time.time()

    transitions = [t for t in incidence_matrix.transitions]

    for a in range(1, k + 1):
        for t in range(len_transitions):
            if transitions[t].label[0] != trace_division[a - 1][0]:
                model.addconstraint(y[a][t] == 0)

    # et_constr5 = time.time()
    # time_constr5 = et_constr5 - st_constr5

    from cvxopt import solvers
    solvers.options['glpk'] = this_options_lp

    # print("EXTENDED STATE EQ SETUP TIME:", timer() - extended_state_eq_setup_time_start)

    extended_state_eq_solve_time_start = timer()
    # st_solve = time.time()
    model.solve(solver='glpk')

    # et_solve = time.time()
    # time_solve = et_solve - st_solve
    # prim_obj corresponds to underestimate h
    prim_obj = model.objective.value()

    prim_obj = prim_obj[0] if prim_obj is not None else sys.maxsize
    # points = points if points is not None else [0.0] * len(sync_net.transitions)

    # points corresponds to solution vector z = x_0 + sum over (x_a + y_a) for 1<= a <= k

    if not model.status == "primal infeasible":

        # solution vector as list of values
        points_list = np.array(x[0].value)
        for a in range(1, k + 1):
            x_np = np.array(x[a].value)
            y_np = np.array(y[a].value)
            points_list = np.add(points_list, (np.add(x_np, y_np)))
        points_list = points_list.flatten()

        # if ilp, check if solution vector is integer and re-compute if not
        if ilp:
            if not check_lp_sol_int(points_list):

                from cvxopt import glpk, blas

                t = model._inmatrixform(format='dense')

                lp1, vmap, mmap = t[0], t[1], t[2]

                variables = lp1.variables()

                x = variables[0]
                c = lp1.objective._linear._coeff[x]

                inequalities = lp1._inequalities

                G = inequalities[0]._f._linear._coeff[x]
                h = -inequalities[0]._f._constant

                equalities = lp1._equalities
                if equalities:
                    A = equalities[0]._f._linear._coeff[x]
                    b = -equalities[0]._f._constant
                else:
                    A = matrix(0.0, (0, len(x)))
                    b = matrix(0.0, (0, 1))

                print("solve ilp")
                size = G.size[1]
                I = {i for i in range(size)}
                status, x = glpk.ilp(c[:], G, h, A, b, I=I, options={"tm_lim": 60000})

                ilp_solved = ilp_solved + 1

                if status == 'optimal':
                    prim_obj = blas.dot(c, x)

                    # de-stack variables according to transitions
                    # points_list = [0.0] * len_transitions
                    # for i in range(size):
                    #   rem = i % len_transitions
                    #  points_list[rem] = points_list[rem] + x[i]

                    points_list = np.array(x)
                    points_list = points_list.reshape((-1, len_transitions))
                    points_list = np.sum(points_list, axis=0)
                else:
                    prim_obj = sys.maxsize

                    points_list = [0.0] * len_transitions
    else:
        points_list = [0.0] * len_transitions

    # print("ilp_solved ", ilp_solved)

    # et_total = time.time()
    # time_total = et_total - st_total

    # print("Time total",time_total)

    # print("EXTENDED STATE EQ SOLVE TIME:", timer() - extended_state_eq_solve_time_start)

    return prim_obj, points_list


def __get_tuple_from_queue(marking, queue):
    for t in queue:
        if t.m == marking:
            return t
    return None


def __vectorize_initial_final_cost(incidence_matrix, ini, fin, cost_function):
    ini_vec = incidence_matrix.encode_marking(ini)
    fini_vec = incidence_matrix.encode_marking(fin)
    cost_vec = [0] * len(cost_function)
    for t in cost_function.keys():
        cost_vec[incidence_matrix.transitions[t]] = cost_function[t]
    return ini_vec, fini_vec, cost_vec


class SearchTuple:
    def __init__(self, f, g, h, m, p, t, x, trust, num_explained = 0):
        self.f = f
        self.g = g
        self.h = h
        self.m = m
        self.p = p
        self.t = t
        self.x = x
        self.trust = trust
        self.num_explained = num_explained

    def __lt__(self, other):
        if not math.isclose(self.f, other.f):
            return self.f < other.f
        elif self.trust ^ other.trust:
            return self.trust
        else:
            return self.h < other.h

    def __get_firing_sequence(self):
        ret = []
        if self.p is not None:
            ret = ret + self.p.__get_firing_sequence()
        if self.t is not None:
            ret.append(self.t)
        return ret

    def __repr__(self):
        string_build = ["\nm=" + str(self.m), " f=" + str(self.f), ' g=' + str(self.g), " h=" + str(self.h),
                        " path=" + str(self.__get_firing_sequence()) + "\n\n"]
        return " ".join(string_build)

class SearchTupleExtStateEq:
    def __init__(self, f, g, h, m, p, t, x, trust, num_explained = 0):
        self.f = f
        self.g = g
        self.h = h
        self.m = m
        self.p = p
        self.t = t
        self.x = x
        self.trust = trust
        self.num_explained = num_explained

    def __lt__(self, other):
        if not math.isclose(self.f, other.f):
            return self.f < other.f
        elif self.trust ^ other.trust:
            return self.trust
        else:
            return self.h < other.h

    def __get_firing_sequence(self):
        ret = []
        if self.p is not None:
            ret = ret + self.p.__get_firing_sequence()
        if self.t is not None:
            ret.append(self.t)
        return ret

    def __repr__(self):
        return self.m.__repr__()


class MappedQueue(object):
    """The MappedQueue class implements an efficient minimum heap. The
    smallest element can be popped in O(1) time, new elements can be pushed
    in O(log n) time, and any element can be removed or updated in O(log n)
    time. The queue cannot contain duplicate elements and an attempt to push an
    element already in the queue will have no effect.

    MappedQueue complements the heapq package from the python standard
    library. While MappedQueue is designed for maximum compatibility with
    heapq, it has slightly different functionality.

    Examples
    --------

    A `MappedQueue` can be created empty or optionally given an array of
    initial elements. Calling `push()` will add an element and calling `pop()`
    will remove and return the smallest element.

    >>> q = MappedQueue([916, 50, 4609, 493, 237])
    >>> q.push(1310)
    True
    >>> x = [q.pop() for i in range(len(q.h))]
    >>> x
    [50, 237, 493, 916, 1310, 4609]

    Elements can also be updated or removed from anywhere in the queue.

    >>> q = MappedQueue([916, 50, 4609, 493, 237])
    >>> q.remove(493)
    >>> q.update(237, 1117)
    >>> x = [q.pop() for i in range(len(q.h))]
    >>> x
    [50, 916, 1117, 4609]

    References
    ----------
    .. [1] Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C. (2001).
       Introduction to algorithms second edition.
    .. [2] Knuth, D. E. (1997). The art of computer programming (Vol. 3).
       Pearson Education.
    """

    def __init__(self, data=[]):
        """Priority queue class with updatable priorities.
        """
        self.h = list(data)
        self.d = dict()
        self._heapify()

    def __len__(self):
        return len(self.h)

    def _heapify(self):
        """Restore heap invariant and recalculate map."""
        heapq.heapify(self.h)
        self.d = dict([(elt.m, pos) for pos, elt in enumerate(self.h)])
        if len(self.h) != len(self.d):
            raise AssertionError("Heap contains duplicate elements")

    def push(self, elt):
        """Add an element to the queue."""
        # If element is already in queue, do nothing
        if elt.m in self.d:
            return False
        # Add element to heap and dict
        pos = len(self.h)
        self.h.append(elt)
        self.d[elt.m] = pos
        # Restore invariant by sifting down
        self._siftdown(pos)
        return True

    def pop(self):
        """Remove and return the smallest element in the queue."""
        # Remove smallest element
        elt = self.h[0]
        del self.d[elt.m]
        # If elt is last item, remove and return
        if len(self.h) == 1:
            self.h.pop()
            return elt
        # Replace root with last element
        last = self.h.pop()
        self.h[0] = last
        self.d[last.m] = 0
        # Restore invariant by sifting up, then down
        pos = self._siftup(0)
        self._siftdown(pos)
        # Return smallest element
        return elt

    def update(self, elt, new):
        """Replace an element in the queue with a new one."""
        # Replace
        pos = self.d[elt.m]
        self.h[pos] = new
        del self.d[elt.m]
        self.d[new.m] = pos
        # Restore invariant by sifting up, then down
        pos = self._siftup(pos)
        self._siftdown(pos)

    def remove(self, elt):
        """Remove an element from the queue."""
        # Find and remove element
        try:
            pos = self.d[elt.m]
            del self.d[elt.m]
        except KeyError:
            # Not in queue
            raise
        # If elt is last item, remove and return
        if pos == len(self.h) - 1:
            self.h.pop()
            return
        # Replace elt with last element
        last = self.h.pop()
        self.h[pos] = last
        self.d[last.m] = pos
        # Restore invariant by sifting up, then down
        pos = self._siftup(pos)
        self._siftdown(pos)

    def _siftup(self, pos):
        """Move element at pos down to a leaf by repeatedly moving the smaller
        child up."""
        h, d = self.h, self.d
        elt = h[pos]
        # Continue until element is in a leaf
        end_pos = len(h)
        left_pos = (pos << 1) + 1
        while left_pos < end_pos:
            # Left child is guaranteed to exist by loop predicate
            left = h[left_pos]
            try:
                right_pos = left_pos + 1
                right = h[right_pos]
                # Out-of-place, swap with left unless right is smaller
                if right < left:
                    h[pos], h[right_pos] = right, elt
                    pos, right_pos = right_pos, pos
                    d[elt.m], d[right.m] = pos, right_pos
                else:
                    h[pos], h[left_pos] = left, elt
                    pos, left_pos = left_pos, pos
                    d[elt.m], d[left.m] = pos, left_pos
            except IndexError:
                # Left leaf is the end of the heap, swap
                h[pos], h[left_pos] = left, elt
                pos, left_pos = left_pos, pos
                d[elt.m], d[left.m] = pos, left_pos
            # Update left_pos
            left_pos = (pos << 1) + 1
        return pos

    def _siftdown(self, pos):
        """Restore invariant by repeatedly replacing out-of-place element with
        its parent."""
        h, d = self.h, self.d
        elt = h[pos]
        # Continue until element is at root
        while pos > 0:
            parent_pos = (pos - 1) >> 1
            parent = h[parent_pos]
            if parent > elt:
                # Swap out-of-place element with parent
                h[parent_pos], h[pos] = elt, parent
                parent_pos, pos = pos, parent_pos
                d[elt.m] = pos
                d[parent.m] = parent_pos
            else:
                # Invariant is satisfied
                break
        return pos


class DijkstraSearchTuple:
    def __init__(self, g, m, p, t, l):
        self.g = g
        self.m = m
        self.p = p
        self.t = t
        self.l = l

    def __lt__(self, other):
        if self.g < other.g:
            return True
        elif other.g < self.g:
            return False
        else:
            return other.l < self.l

    def __get_firing_sequence(self):
        ret = []
        if self.p is not None:
            ret = ret + self.p.__get_firing_sequence()
        if self.t is not None:
            ret.append(self.t)
        return ret

    def __repr__(self):
        string_build = ["\nm=" + str(self.m), " g=" + str(self.g),
                        " path=" + str(self.__get_firing_sequence()) + "\n\n"]
        return " ".join(string_build)


class DijkstraSearchTupleForAntiAndMulti:
    # in this version we keep the run and not the previous element
    # the display is different
    def __init__(self, g, m, r):
        self.g = g
        self.m = m
        self.r = r

    def __lt__(self, other):
        if self.g < other.g:
            return True
        elif other.g < self.g:
            return False
        else:
            return len(other.r) < len(self.r)

    def __get_firing_sequence(self):
        return self.r

    def __repr__(self):
        string_build = ["\nm=" + str(self.m), " g=" + str(self.g),
                        " path=" + str(self.__get_firing_sequence()) + "\n\n"]
        return " ".join(string_build)


class TweakedSearchTuple:
    def __init__(self, f, g, h, m, p, t, x, trust, virgin):
        self.f = f
        self.g = g
        self.h = h
        self.m = m
        self.p = p
        self.t = t
        self.x = x
        self.trust = trust
        # a virgin status must be explored in its firing sequence
        self.virgin = virgin

    def __lt__(self, other):
        if self.f < other.f:
            return True
        elif other.f < self.f:
            return False
        elif self.virgin and not other.virgin:
            return True
        elif self.trust and not other.trust:
            return True
        else:
            return self.h < other.h

    def __get_firing_sequence(self):
        ret = []
        if self.p is not None:
            ret = ret + self.p.__get_firing_sequence()
        if self.t is not None:
            ret.append(self.t)
        return ret

    def __repr__(self):
        string_build = ["\nm=" + str(self.m), " f=" + str(self.f), ' g=' + str(self.g), " h=" + str(self.h),
                        " path=" + str(self.__get_firing_sequence()) + "\n\n"]
        return " ".join(string_build)


def get_visible_transitions_eventually_enabled_by_marking(net, marking):
    """
    Get visible transitions eventually enabled by marking (passing possibly through hidden transitions)
    Parameters
    ----------
    net
        Petri net
    marking
        Current marking
    """
    all_enabled_transitions = sorted(list(semantics.enabled_transitions(net, marking)),
                                     key=lambda x: (str(x.name), id(x)))
    initial_all_enabled_transitions_marking_dictio = {}
    all_enabled_transitions_marking_dictio = {}
    for trans in all_enabled_transitions:
        all_enabled_transitions_marking_dictio[trans] = marking
        initial_all_enabled_transitions_marking_dictio[trans] = marking
    visible_transitions = set()
    visited_transitions = set()

    i = 0
    while i < len(all_enabled_transitions):
        t = all_enabled_transitions[i]
        marking_copy = copy(all_enabled_transitions_marking_dictio[t])

        if repr([t, marking_copy]) not in visited_transitions:
            if t.label is not None:
                visible_transitions.add(t)
            else:
                if semantics.is_enabled(t, net, marking_copy):
                    new_marking = semantics.execute(t, net, marking_copy)
                    new_enabled_transitions = sorted(list(semantics.enabled_transitions(net, new_marking)),
                                                     key=lambda x: (str(x.name), id(x)))
                    for t2 in new_enabled_transitions:
                        all_enabled_transitions.append(t2)
                        all_enabled_transitions_marking_dictio[t2] = new_marking
            visited_transitions.add(repr([t, marking_copy]))
        i = i + 1

    return visible_transitions


def discountedEditDistance(s1, s2, exponent=2, modeled=True):
    '''
    Fast implementation of the discounted distance
    Inspired from the faster version of the edit distance
    '''
    # print(s1,s2)
    if len(s1) < len(s2):
        return discountedEditDistance(s2, s1, exponent=exponent, modeled=False)

    previous_row = [0]
    for a in range(len(s2)):
        if not modeled and (s2[a] == "tau" or s2[a] == None or s2[a][0] == "n"):
            previous_row.append(previous_row[-1])
        else:
            previous_row.append(previous_row[-1] + exponent ** (-(a)))
    for i, c1 in enumerate(s1):
        if modeled:
            exp1 = sum(exponent ** (-(a)) for a in range(i + 1) if s1[a] != "tau" and s1[a] != None and s1[a][0] != "n")
        else:
            exp1 = sum(exponent ** (-(a)) for a in range(i + 1))
        current_row = [exp1]
        for j, c2 in enumerate(s2):

            exp2 = exponent ** (-(i + 1 + j))
            if modeled and (c1 in ["tau", None] or c1[0] == "n" or "skip" in c1):
                insertions = previous_row[
                    j + 1]  # j+1 instead of j since previous_row and current_row are one character longer
                deletions = current_row[j] + exp2  # than s2
            elif not modeled and (c2 in ["tau", None] or c2[0] == "n"):
                insertions = previous_row[
                                 j + 1] + exp2  # j+1 instead of j since previous_row and current_row are one character longer
                deletions = current_row[j]
            else:
                insertions = previous_row[
                                 j + 1] + exp2  # j+1 instead of j since previous_row and current_row are one character longer
                deletions = current_row[j] + exp2
            if (c1 != c2):
                current_row.append(min(insertions, deletions))
            else:
                substitutions = previous_row[j]
                current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row
    return len(s1) + len(s2), previous_row[-1]


def levenshtein(seq1, seq2):
    '''
    Edit distance without substitution
    '''
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y))
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] in [None, "tau"] or seq1[x - 1][0] == 'n' or "skip" in seq1[x - 1] or "tau" in seq1[x - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y],
                    matrix[x, y - 1] + 1
                )
            elif seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x, y - 1] + 1
                )
    return (matrix[size_x - 1, size_y - 1])
