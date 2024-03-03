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
import time
from copy import copy
from typing import List, Tuple

import numpy as np

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

    sol = lp_solver.apply(cost_vec, g_matrix, h_cvx, a_matrix, b_term, parameters=parameters_solving,
                          variant=variant)
    prim_obj = lp_solver.get_prim_obj_from_sol(sol, variant=variant)
    points = lp_solver.get_points_from_sol(sol, variant=variant)

    prim_obj = prim_obj if prim_obj is not None else sys.maxsize
    points = points if points is not None else [0.0] * len(sync_net.transitions)

    return prim_obj, points


def __compute_exact_extended_state_equation(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                            marking, fin_vec, variant, trace_division, use_cvxopt=False, strict=True,
                                            k=1):
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

    return prim_obj, points_list


def check_lp_sol_int(x):
    for i in range(len(x)):
        if abs(x[i] - round(x[i])) > 10 ** (-5):
            return False
    return True


def __compute_exact_extended_state_equation_ilp(sync_net, a_matrix, h_cvx, g_matrix, cost_vec, incidence_matrix,
                                                marking, fin_vec, variant, trace_division, ilp=False, use_cvxopt=False,
                                                strict=True,
                                                k=1):
    # ilp version

    # trace_divison = [substring_1, ..., substring_k]

    # return solution vector z = x_0 + sum ( x_a + y_a )
    # return estimate h
    #st_total = time.time()

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
    #cost_transp = np.transpose(cost_vec)

    #st_obj = time.time()

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

    #et_obj = time.time()
    #time_obj = et_obj - st_obj
    # c1: marking eq., reach final marking from initial/current after combining all firing transitions of all x_a and y_a
    # was, wenn mittendrin berechnet wird? Trotzdem von initial marking?

    #st_constr1 = time.time()

    m_vec = incidence_matrix.encode_marking(marking)
    #m_vec_2d = [m_vec[i:i + 1] for i in range(0, len(m_vec), 1)]
    m_vec_2d = [m_vec[i:i + 1] for i in range(len_places)]
    m_matrix = matrix(m_vec_2d, (len_places, 1), tc='d')

    #fin_vec_2d = [fin_vec[i:i + 1] for i in range(0, len(fin_vec), 1)]
    fin_vec_2d = [fin_vec[i:i + 1] for i in range(len_places)]
    fin_matrix = matrix(fin_vec_2d, (len_places, 1), tc='d')

    sums_transitions = x[1] + y[1]

    for i in range(2, k + 1):
        sums_transitions = sums_transitions + x[i] + y[i]
    # op.addconstraint(m_matrix + matrix(a_matrix) * x[0] + matrix(a_matrix) * sums_transitions == fin_matrix)

    m_a_matrix = matrix(a_matrix, tc='d')

    constr1 = (m_matrix + m_a_matrix * x[0] + m_a_matrix * sums_transitions == fin_matrix)
    model.addconstraint(constr1)

    #et_constr1 = time.time()
    #time_constr1 = et_constr1 - st_constr1

    # c2: extended marking eq., after firing prefix of transitions, sufficient tokens available to fire first transition in y_a
    #st_constr2 = time.time()

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

    #et_constr2 = time.time()
    #time_constr2 = et_constr2 - st_constr2

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

    #st_constr5 = time.time()

    transitions = [t for t in incidence_matrix.transitions]

    for a in range(1, k + 1):
        for t in range(len_transitions):
            if transitions[t].label[0] != trace_division[a - 1][0]:
                model.addconstraint(y[a][t] == 0)

    #et_constr5 = time.time()
    #time_constr5 = et_constr5 - st_constr5

    from cvxopt import solvers
    solvers.options['glpk'] = this_options_lp

    #st_solve = time.time()
    model.solve(solver='glpk')

    #et_solve = time.time()
    #time_solve = et_solve - st_solve
    # prim_obj corresponds to underestimate h
    prim_obj = model.objective.value()

    prim_obj = prim_obj[0] if prim_obj is not None else sys.maxsize
    # points = points if points is not None else [0.0] * len(sync_net.transitions)

    # points corresponds to solution vector z = x_0 + sum over (x_a + y_a) for 1<= a <= k

    if not model.status == "primal infeasible":

        # solution vector as list of values
        points_list = np.array(x[0].value)
        for a in range(1, k+1):
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
                    #points_list = [0.0] * len_transitions
                    #for i in range(size):
                     #   rem = i % len_transitions
                      #  points_list[rem] = points_list[rem] + x[i]

                    points_list = np.array(x)
                    points_list = points_list.reshape((-1, len_transitions))
                    points_list = np.sum(points_list, axis =0)
                else:
                    prim_obj = sys.maxsize

                    points_list = [0.0] * len_transitions
    else:
        points_list = [0.0] * len_transitions

    #print("ilp_solved ", ilp_solved)

    #et_total = time.time()
    #time_total = et_total - st_total

    #print("Time total",time_total)

    return prim_obj, points_list, ilp_solved


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
    def __init__(self, f, g, h, m, p, t, x, trust):
        self.f = f
        self.g = g
        self.h = h
        self.m = m
        self.p = p
        self.t = t
        self.x = x
        self.trust = trust

    def __lt__(self, other):
        if self.f < other.f:
            return True
        elif other.f < self.f:
            return False
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
