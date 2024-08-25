import collections

import pandas as pd
import numpy as np
import time
import random
import math
import matplotlib.pyplot as plt

# plt.use("pgf")
plt.rcParams.update({
    # "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    # 'text.usetex': True,
    # 'pgf.rcfonts': False,
    'font.size': 11
})

from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
import seaborn as sns
from timeit import default_timer as timer

import resource
import sys

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier

# from imblearn.over_sampling import RandomOverSampler
# from imblearn.under_sampling import RandomUnderSampler
# from collections import Counter

import statistical_numbers
import features
import plot_data


# import preprocess_data


# data = merge_data.get_merged_data()

# data = pd.read_pickle("results/road_heuristic_3.pkl")

# plot_data.create_sunburst_plot(data, 180)

# Encode data
# Schnellste Heuristic Time = 1, else 0

# multi class classification
# label column
def get_merged_data():
    # under_sample road_inductive_0 = pd.read_pickle("results/road_inductive_0_3.pkl")
    road_inductive_02 = pd.read_pickle("results/road_inductive_02_updated_eqs.pkl")  # up
    road_heuristic = pd.read_pickle("results/road_heuristics_updated_eqs.pkl")  # up
    # road_filtered = pd.read_pickle("results/road_filtered_inductive_0_3.pkl")
    # road_max_cov = pd.read_pickle("results/road_max_cov_filtered_inductive_0_3.pkl")
    road_top_10 = pd.read_pickle("results/road_top_10_updated_eqs.pkl")  # up
    road_top_5 = pd.read_pickle("results/road_top_5_updated_eqs.pkl")  # up
    road_top_1 = pd.read_pickle("results/road_top_1_updated_eqs.pkl")  # up

    # sepsis_alpha = pd.read_pickle("results/sepsis_alpha_updated_ext.pkl")  # up
    sepsis_inductive_02 = pd.read_pickle("results/sepsis_inductive_02_updated_eqs.pkl")  # up
    # sepsis_filtered = pd.read_pickle("results/sepsis_filtered_inductive_0_3.pkl")
    # sepsis_max_cov = pd.read_pickle("results/sepsis_max_cov_filtered_inductive_0_3.pkl")
    # sepsis_top_10 = pd.read_pickle("results/sepsis_top_10_inductive_0_curr.pkl")  # 343
    # sepsis_top_5 = pd.read_pickle("results/sepsis_top_5_updated_ext.pkl")  # up
    sepsis_top_1 = pd.read_pickle("results/sepsis_top_1_updated_eqs.pkl")  # up

    # italian_alpha = pd.read_pickle("results/italian_alpha_0.2_3.pkl")

    # bpi12
    # under_sample bpi12_inductive_02_curr = pd.read_pickle("results/bpi12_inductive_0.2_curr.pkl") lohnt nicht
    # bpi12_filtered = pd.read_pickle("results/bpi12_filtered_inductive_0_3.pkl")
    # bpi12_max_cov = pd.read_pickle("results/bpi12_max_cov_inductive_0_curr.pkl")
    # bpi12_top_10 = pd.read_pickle("results/bpi12_top_10_inductive_0_curr.pkl")  # 1099?
    # bpi12_ext_top_5 1361
    bpi12_top_1 = pd.read_pickle("results/bpi12_top_1_updated_eqs.pkl")  # up
    # bpi13
    prGm6 = pd.read_pickle("results/prGm6_updated_eqs.pkl")  # up
    prFm6 = pd.read_pickle("results/prFm6_updated_eqs.pkl")  # up

    # BPI20
    # under_sample prepaid_inductive_0 = pd.read_pickle("results/prepaid_inductive_0_3.pkl")
    prepaid_inductive_02 = pd.read_pickle("results/prepaid_inductive_02_updated_eqs.pkl")
    # prepaid_filtered = pd.read_pickle("results/prepaid_filtered_inductive_0_3.pkl")
    # prepaid_max_cov = pd.read_pickle("results/prepaid_max_cov_filtered_inductive_0_3.pkl")
    prepaid_top_10 = pd.read_pickle("results/prepaid_top_10_updated_eqs.pkl")  # up
    prepaid_top_5 = pd.read_pickle("results/prepaid_top_5_updated_eqs.pkl")  # up
    prepaid_top_1 = pd.read_pickle("results/prepaid_top_1_updated_eqs.pkl")  # up

    # under_sample request_inductive_0 = pd.read_pickle("results/request_inductive_0_3.pkl")
    request_inductive_02 = pd.read_pickle("results/request_inductive_02_updated_eqs.pkl")
    # request_filtered = pd.read_pickle("results/request_filtered_inductive_0_3.pkl")
    # request_max_cov = pd.read_pickle("results/request_max_cov_filtered_inductive_0_3.pkl")
    request_top_10 = pd.read_pickle("results/request_top_10_updated_eqs.pkl")  # up
    request_top_5 = pd.read_pickle("results/request_top_5_updated_eqs.pkl")  # up
    request_top_1 = pd.read_pickle("results/request_top_1_updated_eqs.pkl")  # up

    # under_sample domestic_inductive_0 = pd.read_pickle("results/domestic_inductive_0_3.pkl")
    domestic_inductive_02 = pd.read_pickle("results/domestic_inductive_02_updated_eqs.pkl")
    # domestic_filtered = pd.read_pickle("results/domestic_filtered_inductive_0_3.pkl")
    # domestic_max_cov = pd.read_pickle("results/domestic_max_cov_filtered_inductive_0_3.pkl")
    domestic_top_10 = pd.read_pickle("results/domestic_top_10_updated_eqs.pkl")  # up
    domestic_top_5 = pd.read_pickle("results/domestic_top_5_updated_eqs.pkl")  # up
    domestic_top_1 = pd.read_pickle("results/domestic_top_1_updated_eqs.pkl")  # up

    # under_sample international_inductive_0 = pd.read_pickle("results/international_declaration_inductive_0_3.pkl")
    international_inductive_02 = pd.read_pickle("results/international_inductive_02_updated_eqs.pkl")
    # international_filtered = pd.read_pickle("results/international_declaration_filtered_inductive_0_3.pkl")
    # international_max_cov = pd.read_pickle("results/international_declaration_max_cov_filtered_inductive_0_3.pkl")
    international_top_10 = pd.read_pickle("results/international_top_10_updated_eqs.pkl")  # up
    international_top_5 = pd.read_pickle("results/international_top_5_updated_eqs.pkl")  # up
    international_top_1 = pd.read_pickle("results/international_top_1_updated_eqs.pkl")  # up

    # permit_inductive_0 = pd.read_pickle("results/permit_inductive_0_curr.pkl") not found?
    # under_sample permit_inductive_02 = pd.read_pickle("permit_inductive_0.2_curr.pkl")
    # permit_filtered_curr = pd.read_pickle("results/permit_inductive_0_curr.pkl") not found?
    # permit_top_5 = pd.read_pickle("results/permit_top_5_inductive_0.pkl")

    data = road_inductive_02
    # under_sample data = data._append(road_inductive_0)
    data = data._append(road_heuristic)
    # data = data._append(road_filtered)
    # data = data._append(road_max_cov)
    data = data._append(road_top_10)
    data = data._append(road_top_5)
    data = data._append(road_top_1)

    # data = data._append(sepsis_alpha)
    data = data._append(sepsis_inductive_02)
    # data = data._append(sepsis_filtered)
    # data = data._append(sepsis_max_cov)
    # data = data._append(sepsis_top_10)
    # data = data._append(sepsis_top_5)
    data = data._append(sepsis_top_1)

    # data = data._append(italian_alpha)

    # under_sample data = data._append(bpi12_inductive_02_curr)
    # data = data._append(bpi12_filtered)
    # data = data._append(bpi12_max_cov)
    # data = data._append(bpi12_top_10)
    data = data._append(bpi12_top_1)

    data = data._append(prGm6)
    data = data._append(prFm6)

    # under_sample data = data._append(prepaid_inductive_0)
    data = data._append(prepaid_inductive_02)
    # data = data._append(prepaid_filtered)
    # data = data._append(prepaid_max_cov)
    data = data._append(prepaid_top_10)
    data = data._append(prepaid_top_5)
    data = data._append(prepaid_top_1)

    # under_sample data = data._append(request_inductive_0)
    data = data._append(request_inductive_02)
    # data = data._append(request_filtered)
    # data = data._append(request_max_cov)
    data = data._append(request_top_10)
    data = data._append(request_top_5)
    data = data._append(request_top_1)

    # under_sample data = data._append(domestic_inductive_0)
    data = data._append(domestic_inductive_02)
    # data = data._append(domestic_filtered)
    # data = data._append(domestic_max_cov)
    data = data._append(domestic_top_10)
    data = data._append(domestic_top_5)
    data = data._append(domestic_top_1)

    # under_sample data = data._append(international_inductive_0)
    data = data._append(international_inductive_02)
    # data = data._append(international_filtered)
    # data = data._append(international_max_cov)
    data = data._append(international_top_10)
    data = data._append(international_top_5)
    data = data._append(international_top_1)

    # data = data._append(permit_inductive_0)
    # under_sample data = data._append(permit_inductive_02)
    # data = data._append(permit_filtered_curr)
    # data = data._append(permit_top_5)

    data = data.reset_index(drop=True)

    return data


def create_features(trace, pn, im, fm):
    curr_features = []

    start_feat = timer()
    _, visited, queued, deadlock, boundedness = features.random_playout(pn, im, fm, 10, 50)
    feat_visited_queued_deadlock_boundedness = timer() - start_feat

    start_feat = timer()
    matches_traces, matches_traces_ratio, matches_models, matches_models_ratio = features.matching_labels(pn, trace)
    feat_matching_labels = timer() - start_feat

    curr_features.append(matches_traces)
    curr_features.append(matches_traces_ratio)
    curr_features.append(matches_models)
    curr_features.append(matches_models_ratio)

    start_feat = timer()
    matching_starts = features.matching_starts(pn, trace)
    feat_matching_start = timer() - start_feat
    curr_features.append(matching_starts)

    start_feat = timer()
    matching_ends = features.matching_ends(pn, trace)
    feat_matching_end = timer() - start_feat
    curr_features.append(matching_ends)

    start_feat = timer()
    len_trace, trace_transitions_ratio, trace_places_ratio, transitions_trace_ratio, places_trace_ratio = features.trace_ratio(
        pn, trace)
    feat_trace_len = timer() - start_feat
    curr_features.append(len_trace)
    curr_features.append(trace_transitions_ratio)
    curr_features.append(trace_places_ratio)
    curr_features.append(transitions_trace_ratio)
    curr_features.append(places_trace_ratio)

    start_feat = timer()
    distinct_events = features.distinct_events_trace(trace)
    feat_distinc_events = timer() - start_feat
    curr_features.append(distinct_events)

    start_feat = timer()
    trace_loop, trace_loop_ratio, max_reps, max_reps_ratio, mean_reps, mean_reps_ratio, sum_reps, sum_reps_ratio = (
        features.trace_loop(trace))
    feat_trace_loop = timer() - start_feat
    curr_features.append(trace_loop)
    curr_features.append(trace_loop_ratio)
    curr_features.append(max_reps)
    curr_features.append(max_reps_ratio)
    curr_features.append(mean_reps)
    curr_features.append(mean_reps_ratio)
    curr_features.append(sum_reps)
    curr_features.append(sum_reps_ratio)

    start_feat = timer()
    one_length_loop = features.one_length_loop(trace)
    feat_one_lenght_loop = timer() - start_feat
    curr_features.append(one_length_loop)

    start_feat = timer()
    model_duplicates, model_duplicates_ratio = features.model_duplicates(pn)
    feat_model_duplicates = timer() - start_feat
    curr_features.append(model_duplicates)
    curr_features.append(model_duplicates_ratio)

    start_feat = timer()
    trans_no_in_arc, trans_no_in_arc_ratio = features.transitions_no_in_arc(pn)
    feat_trans_no_in_arc = timer() - start_feat
    curr_features.append(trans_no_in_arc)
    curr_features.append(trans_no_in_arc_ratio)

    start_feat = timer()
    silent_transitions, silent_transitions_ratio = features.model_silent_transitions(pn)
    feat_silent_trans = timer() - start_feat
    curr_features.append(silent_transitions)
    curr_features.append(silent_transitions_ratio)

    start_feat = timer()
    parallelism_sum, parallelism_ratio = features.parallelism(pn)
    feat_parallelism_sum = timer() - start_feat
    curr_features.append(parallelism_sum)
    curr_features.append(parallelism_ratio)

    start_feat = timer()
    parallelism_mult, parallelism_mult_ratio = features.parallelism_model_multiplied(pn)
    feat_parallelism_mult = timer() - start_feat
    curr_features.append(parallelism_mult)
    curr_features.append(parallelism_mult_ratio)

    start_feat = timer()
    choice_sum, choice_ratio, choice_mult, choice_mult_ratio = features.choice(pn)
    feat_choice = timer() - start_feat
    curr_features.append(choice_sum)
    curr_features.append(choice_ratio)
    curr_features.append(choice_mult)
    curr_features.append(choice_mult_ratio)

    start_feat = timer()
    curr_features.append(simplicity_evaluator.apply(pn))
    feat_simplicity = timer() - start_feat

    start_feat = timer()
    free_choice = features.free_choice(pn)
    feat_free_choice = timer() - start_feat
    curr_features.append(free_choice)

    curr_features.append(visited)
    curr_features.append(queued)
    curr_features.append(deadlock)
    curr_features.append(boundedness)

    times_features = {"Visited, Queued, Deadlocks, Boundedness": feat_visited_queued_deadlock_boundedness,
                      "Matching Labels": feat_matching_labels,
                      "Matching Start": feat_matching_start, "Matching End": feat_matching_end,
                      "Trace Length": feat_trace_len,
                      "Distinct Events": feat_distinc_events, "Trace Loop": feat_trace_loop,
                      "Trace One-Length Loop": feat_one_lenght_loop,
                      "Model Duplicates": feat_model_duplicates, "Transitions no in-arc": feat_trans_no_in_arc,
                      "Silent Transitions": feat_silent_trans,
                      "Parallelism Sum": feat_parallelism_sum, "Parallelism Mult": feat_parallelism_mult,
                      "Choice": feat_choice,
                      "Simplicity": feat_simplicity, "Free choice": feat_free_choice
                      }

    return curr_features, times_features


# data = merge_data.get_merged_data()

def label_data(data):
    # Encode data
    # Schnellste Heuristic Time = 1, else 0

    # multi class classification
    # label column
    plot_data.create_sunburst_plot(data, 180, "total")

    min_val_idx = data[["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time",
                        "Ext. Eq. LP Time", "Ext. Eq. ILP Time"]].idxmin(axis="columns")

    encode_times = data.copy()
    labels = []

    row_num = 0

    timeout = 180

    for i in min_val_idx:
        row = encode_times.iloc[row_num, :]
        min_time = row[i]

        if min_time > timeout:
            labels.append("Timeout")
        else:
            if i == "No Heuristic Time":
                labels.append("No Heuristic")
            elif i == "Naive Time":
                labels.append("Naive")
            elif i == "State Eq. LP Time":
                labels.append("State Eq. LP")
            elif i == "State Eq. ILP Time":
                # labels.append("State Eq. ILP")
                # label as State Eq. LP
                labels.append("State Eq. LP")
            elif i == "Ext. Eq. LP Time":
                labels.append("Ext. Eq. LP")
            elif i == "Ext. Eq. ILP Time":
                # labels.append("Ext. Eq. ILP")
                # label as State Eq. LP
                labels.append("Ext. Eq. LP")

        row_num = row_num + 1

    encode_times.insert(0, "Label", labels)

    # remove rows encoded with timeout
    encode_times_drop_timeout = encode_times[encode_times.Label != "Timeout"]

    # one_hot_encoded = pd.get_dummies(encode_times, columns=["Label"])
    return encode_times_drop_timeout.reset_index(drop=True)


def multi_label_data(data):
    """
    label data with list of fastest heuristic
    Parameters
    ----------
    data

    Returns
    -------

    """
    min_val_idx = data[["No Heuristic Time", "Naive Time", "State Eq. LP Time",
                        "Ext. Eq. LP Time"]].idxmin(axis="columns")

    encode_times = data.copy()
    labels = []

    row_num = 0
    timeout = 180

    for i in min_val_idx:
        row = encode_times.iloc[row_num, :]
        min_time = row[i]

        curr_labels = []

        if min_time > timeout:
            labels.append("Timeout")
        else:
            times = {"No Heuristic": row["No Heuristic Time"], "Naive": row["Naive Time"],
                     "State Eq. LP": row["State Eq. LP Time"],
                     # min(row["State Eq. LP Time"], row["State Eq. ILP Time"]),
                     "Ext. Eq. LP": row["Ext. Eq. LP Time"]
                     # min(row["Ext. Eq. LP Time"], row["Ext. Eq. ILP Time"])
                     }

            curr_labels = [key for key, value in times.items() if value <= (1.1 * min_time)]

            labels.append(curr_labels)

        row_num += 1

    encode_times.insert(0, "Label Multi", labels)
    # remove rows encoded with timeout
    encode_times_drop_timeout = encode_times[encode_times.Label != "Timeout"]

    return encode_times_drop_timeout.reset_index(drop=True)


def sample_data(data, drop_frac=.5):
    data = data.drop(data.query('Label == "State Eq. LP"').sample(frac=drop_frac).index)
    plot_data.create_sunburst_plot(data, 180, "sampled_2")

    return data.reset_index(drop=True)


def rank_heuristics(data, use_ilp=True, reverse=False, column_no="No Heuristic Time", column_naive="Naive Time",
                    column_state_lp="State Eq. LP Time", column_state_ilp="State Eq. ILP Time",
                    column_ext_lp="Ext. Eq. LP Time", column_ext_ilp="Ext. Eq. ILP Time"):
    encode_ranks = data.copy()

    no_rank = []
    naive_rank = []
    state_lp_rank = []
    ext_lp_rank = []

    if use_ilp:
        state_ilp_rank = []
        ext_ilp_rank = []

    for i in encode_ranks.index:
        row = data.iloc[i, :]

        # times = []
        if use_ilp:
            no_time = row[column_no]
            naive_time = row[column_naive]
            state_lp_time = row[column_state_lp]
            state_ilp_time = row[column_state_ilp]
            ext_lp_time = row[column_ext_lp]
            ext_ilp_time = row[column_ext_ilp]

        else:
            no_time = row[column_no]
            naive_time = row[column_naive]
            state_lp_time = row[column_state_lp]
            ext_lp_time = row[column_ext_lp]

        if use_ilp:
            heuristics = ["No Heuristic", "Naive", "State LP", "Ext. LP", "State ILP", "Ext. ILP"]
            times_dict = {"No Heuristic": no_time, "Naive": naive_time, "State LP": state_lp_time,
                          "State ILP": state_ilp_time, "Ext. LP": ext_lp_time, "Ext. ILP": ext_ilp_time}
        else:
            heuristics = ["No Heuristic", "Naive", "State LP", "Ext. LP"]
            times_dict = {"No Heuristic": no_time, "Naive": naive_time, "State LP": state_lp_time,
                          "Ext. LP": ext_lp_time}

        sorted_keys = {k: v for k, v in sorted(times_dict.items(), key=lambda x: x[1], reverse=reverse)}
        sorted_keys_ls = [k for k, v in sorted_keys.items()]

        ranks = []
        for h in heuristics:
            ranks.append(sorted_keys_ls.index(h) + 1)

        no_rank.append(ranks[0])
        naive_rank.append(ranks[1])
        state_lp_rank.append(ranks[2])
        ext_lp_rank.append(ranks[3])

        if use_ilp:
            state_ilp_rank.append(ranks[4])
            ext_ilp_rank.append(ranks[5])

    encode_ranks.insert(1, "No Heuristic Rank", no_rank)
    encode_ranks.insert(2, "Naive Rank", naive_rank)
    encode_ranks.insert(3, "State LP Rank", state_lp_rank)
    encode_ranks.insert(4, "Ext. LP Rank", ext_lp_rank)

    if use_ilp:
        encode_ranks.insert(5, "State ILP Rank", state_ilp_rank)
        encode_ranks.insert(6, "Ext. ILP Rank", ext_ilp_rank)

    return encode_ranks


def get_model_features(pn, im, fm):
    start_time = time.time()
    curr_features = []
    _, visited, queued, deadlock, boundedness = features.random_playout(pn, im, fm, 10, 50)

    model_duplicates, model_duplicates_ratio = features.model_duplicates(pn)
    curr_features.append(model_duplicates)
    curr_features.append(model_duplicates_ratio)

    trans_no_in_arc, trans_no_in_arc_ratio = features.transitions_no_in_arc(pn)
    curr_features.append(trans_no_in_arc)
    curr_features.append(trans_no_in_arc_ratio)

    silent_transitions, silent_transitions_ratio = features.model_silent_transitions(pn)
    curr_features.append(silent_transitions)
    curr_features.append(silent_transitions_ratio)

    parallelism_sum, parallelism_ratio = features.parallelism(pn)
    curr_features.append(parallelism_sum)
    curr_features.append(parallelism_ratio)

    parallelism_mult, parallelism_mult_ratio = features.parallelism_model_multiplied(pn)
    curr_features.append(parallelism_mult)
    curr_features.append(parallelism_mult_ratio)

    choice_sum, choice_ratio, choice_mult, choice_mult_ratio = features.choice(pn)
    curr_features.append(choice_sum)
    curr_features.append(choice_ratio)
    curr_features.append(choice_mult)
    curr_features.append(choice_mult_ratio)

    curr_features.append(simplicity_evaluator.apply(pn))

    free_choice = features.free_choice(pn)
    curr_features.append(free_choice)

    curr_features.append(visited)
    curr_features.append(queued)
    curr_features.append(deadlock)
    curr_features.append(boundedness)

    time_model_features = time.time() - start_time

    return curr_features, time_model_features


def get_trace_features(trace, pn):
    curr_features = []

    start_feat = timer()
    len_trace, trace_transitions_ratio, trace_places_ratio, transitions_trace_ratio, places_trace_ratio = features.trace_ratio(
        pn, trace)
    feat_trace_len = timer() - start_feat
    curr_features.append(len_trace)
    curr_features.append(trace_transitions_ratio)
    curr_features.append(trace_places_ratio)
    curr_features.append(transitions_trace_ratio)
    curr_features.append(places_trace_ratio)

    start_feat = timer()
    distinct_events = features.distinct_events_trace(trace)
    feat_distinc_events = timer() - start_feat
    curr_features.append(distinct_events)

    start_feat = timer()
    trace_loop, trace_loop_ratio, max_reps, max_reps_ratio, mean_reps, mean_reps_ratio, sum_reps, sum_reps_ratio = (
        features.trace_loop(trace))
    feat_trace_loop = timer() - start_feat
    curr_features.append(trace_loop)
    curr_features.append(trace_loop_ratio)
    curr_features.append(max_reps)
    curr_features.append(max_reps_ratio)
    curr_features.append(mean_reps)
    curr_features.append(mean_reps_ratio)
    curr_features.append(sum_reps)
    curr_features.append(sum_reps_ratio)

    start_feat = timer()
    one_length_loop = features.one_length_loop(trace)
    feat_one_lenght_loop = timer() - start_feat
    curr_features.append(one_length_loop)

    return curr_features


def get_model_trace_features(pn, trace):
    curr_features = []

    start_feat = timer()
    matches_traces, matches_traces_ratio, matches_models, matches_models_ratio = features.matching_labels(pn, trace)
    feat_matching_labels = timer() - start_feat

    curr_features.append(matches_traces)
    curr_features.append(matches_traces_ratio)
    curr_features.append(matches_models)
    curr_features.append(matches_models_ratio)

    start_feat = timer()
    matching_starts = features.matching_starts(pn, trace)
    feat_matching_start = timer() - start_feat
    curr_features.append(matching_starts)

    start_feat = timer()
    matching_ends = features.matching_ends(pn, trace)
    feat_matching_end = timer() - start_feat
    curr_features.append(matching_ends)

    return curr_features


def get_features_data(data):
    data = data.reset_index(drop=True)
    # features_data = []
    feature = []

    # start_features = time.time()
    time_features = 0
    time_model_features = 0

    dict_times_features_ls = []

    dict_pn_id = {}

    count = 0
    for i in data.index:
        row = data.iloc[i, :]
        trace = row["Trace"]
        pn = row["Petri Net"]
        im = row["Initial Marking"]
        fm = row["Final Marking"]

        pn_id = str(id(pn))

        if pn_id not in dict_pn_id:
            model_features, curr_time_model_features = get_model_features(pn, im, fm)
            dict_pn_id[pn_id] = model_features
            time_model_features += curr_time_model_features

        else:
            model_features = dict_pn_id[pn_id]

        # compute other features
        time_start = time.time()
        trace_features = get_trace_features(trace, pn)
        model_trace_features = get_model_trace_features(pn, trace)
        time_features += time.time() - time_start

        row_features = model_trace_features + trace_features + model_features

        #start_time_features = time.time()
        #row_features, dict_times_features = create_features(trace, pn, im, fm)
        #time_features += time.time() - start_time_features

        len_features = len(row_features)

        feature.append(row_features)

        #dict_times_features_ls.append(dict_times_features)
        count += 1

    #counter = collections.Counter()
    #for d in dict_times_features_ls:
     #   counter.update(d)

    #counter = dict(counter)
    #counter.update((x, y / count) for x, y in counter.items())

    #sorted_keys_times_features = {k: v for k, v in sorted(counter.items(), key=lambda x: x[1])}
    #print("Time to compute features in ascending order:")
    #print(sorted_keys_times_features)

    # end_features = time.time()
    # time_features = end_features - start_features
    print("Unique petri nets: ", len(dict_pn_id))
    print("Time to compute model features: ", time_model_features)

    print("Time to compute other features: ", time_features)
    print("Time to compute all features: ", time_features + time_model_features)

    # features_data.append(feature)

    columns = ["Matching Labels Trace", "Matching Trace Ratio", "Matching Labels Models",
               "Matching Model Ratio", "Matching Starts", "Matching Ends",
               "Trace Length", "Trace Trans Ratio",
               "Trace Place Ratio", "Trans Trace Ratio", "Place Trace Ratio", "Distinct Events",
               "Trace Loop", "Trace Loop Ratio", "Trace Loop Max",
               "Trace Loop Max Ratio", "Trace Loop Mean", "Trace Loop Mean Ratio",
               "Trace Loop Sum", "Trace Loop Sum Ratio", "Trace One-Length Loop",
               "Model Duplicates", "Model Duplicates Ratio",
               "Trans No In-arc", "Trans No In-arc ratio",
               "Silent Transitions", "Silent Transitions ratio",
               "Parallelism Sum",
               "Parallelism Sum Ratio", "Parallelism Mult", "Parallelism Mult Ratio",
               "Choice Sum", "Choice Sum Ratio", "Choice Mult", "Choice Mult Ratio",
               "Simplicity", "Free-Choice",
               "Visited States", "Queued States", "Deadlocks", "Boundedness"
               ]

    features_df = pd.DataFrame(feature, columns=columns, dtype=float)

    # scale features
    #scaler = MinMaxScaler()
    #scaler = StandardScaler()
    scaler = MaxAbsScaler()
    scaled_features = scaler.fit_transform(features_df)
    scaled_df = pd.DataFrame(scaled_features, columns=features_df.columns)

    result = data.join(scaled_df)

    print(result)
    return result, len_features, time_features + time_model_features


def get_train_test_set(data, ratio=0.8, multi_label=False):
    for i in range(len(data)):

        # create train test split
        data = data.sample(frac=1, random_state=i)  # shuffle data

        total_rows = data.shape[0]
        train_size = int(total_rows * ratio)

        # Split data into test and train
        train = data[0:train_size]
        test = data[train_size:]

        X_train = train.iloc[:, 1:]
        # y = np.asarray(result.iloc[:, :1], dtype=object)
        y_train = train.iloc[:, :1]
        X_test = test.iloc[:, 1:]
        y_test = test.iloc[:, :1]

        label_column = y_train.columns[0]

        y_train_ls = y_train[label_column].to_list()
        y_test_ls = y_test[label_column].to_list()

        if multi_label:
            y_train_ls = [element for innerList in y_train_ls for element in innerList]
            y_test_ls = [element for innerList in y_test_ls for element in innerList]

        if len(set(y_train_ls)) == len(set(y_test_ls)):
            print("Seed for train-test split", i)
            return X_train, y_train, X_test, y_test
    """
    data = data.sample(frac=1, random_state=5)  # shuffle data

    total_rows = data.shape[0]
    train_size = int(total_rows * ratio)

    # Split data into test and train
    train = data[0:train_size]
    test = data[train_size:]

    X_train = train.iloc[:, 1:]
    # y = np.asarray(result.iloc[:, :1], dtype=object)
    y_train = train["Label"]
    X_test = test.iloc[:, 1:]
    y_test = test["Label"]

    return X_train, y_train, X_test, y_test
    """


def ranking_accuracy_top_k(df_true, df_proba, k, use_ilp=False):
    """
    Computes the score of the optimal heuristic in the top k predictions
    Parameters
    ----------
    df_true
    df_proba
    k
    use_ilp

    Returns
    -------

    """

    num_in_top_k = 0
    total = len(df_true)
    for i in range(total):
        row_true = df_true.iloc[i, :]
        row_predict = df_proba.iloc[i, :]
        if not use_ilp:
            ranks_true = {"No Heuristic": row_true["No Heuristic Rank"], "Naive": row_true["Naive Rank"],
                          "State LP": row_true["State LP Rank"], "Ext. LP": row_true["Ext. LP Rank"]}
            ranks_predict = {"No Heuristic": row_predict["No Heuristic Rank"], "Naive": row_predict["Naive Rank"],
                             "State LP": row_predict["State LP Rank"], "Ext. LP": row_predict["Ext. LP Rank"]}

        else:
            ranks_true = {"No Heuristic": row_true["No Heuristic Rank"], "Naive": row_true["Naive Rank"],
                          "State LP": row_true["State LP Rank"], "Ext. LP": row_true["Ext. LP Rank"],
                          "State ILP": row_true["State ILP Rank"], "Ext. ILP": row_true["Ext. ILP Rank"]}
            ranks_predict = {"No Heuristic": row_predict["No Heuristic Rank"], "Naive": row_predict["Naive Rank"],
                             "State LP": row_predict["State LP Rank"], "Ext. LP": row_predict["Ext. LP Rank"],
                             "State ILP": row_predict["State ILP Rank"], "Ext. ILP": row_predict["Ext. ILP Rank"]}

        sorted_keys_true = {k: v for k, v in sorted(ranks_true.items(), key=lambda x: x[1])}
        sorted_ls_true = [k for k, v in sorted_keys_true.items()]

        sorted_keys_proba = {k: v for k, v in sorted(ranks_predict.items(), key=lambda x: x[1])}
        sorted_ls_proba = [k for k, v in sorted_keys_proba.items()]

        for j in range(k):
            if sorted_ls_true[0] == sorted_ls_proba[j]:
                num_in_top_k += 1
                break

    return num_in_top_k / total


def mean_reciprocal_rank():
    pass


def recommendation_function(df_train: object, y_train: object, df_test: object, df_proba: object, max_iter: object,
                            k: object, n: object, max_difference: object, prob_threshold,
                            weight: object = "majority") -> object:
    # max_difference ~ n * 0.025
    # join df_test with df_proba
    df_test_copy = df_test.copy()
    df_test_copy = df_test_copy.reset_index()
    test_proba = df_test_copy.join(df_proba, lsuffix="_alignment")
    df_train = df_train.reset_index()
    y_train = y_train.reset_index()

    recommendation_ls = []
    majority_ambiguous = 0
    # if probability estimates < threshold
    count_do_k_neighbour = 0
    for i in test_proba.index:

        row = test_proba.iloc[i, :]

        prob_dens = {"No Heuristic": row["No Heuristic"], "Naive": row["Naive"], "State Eq. LP": row["State Eq. LP"],
                     "Ext. Eq. LP": row["Ext. Eq. LP"]}

        max_prob = max(prob_dens.values())

        if max_prob > prob_threshold:
            prediction = max(prob_dens, key=lambda x: prob_dens[x])
            recommendation_ls.append(prediction)
            continue

        count_do_k_neighbour += 1

        # retrieve features of test
        features = ["Matching Labels Trace", "Matching Trace Ratio", "Matching Labels Models", "Matching Model Ratio",
                    "Matching Starts", "Matching Ends", "Trace Length", "Trace Trans Ratio", "Trace Place Ratio",
                    "Trans Trace Ratio", "Place Trace Ratio", "Distinct Events", "Trace Loop", "Trace Loop Ratio",
                    "Trace Loop Max", "Trace Loop Max Ratio", "Trace Loop Mean", "Trace Loop Mean Ratio",
                    "Trace Loop Sum", "Trace Loop Sum Ratio", "Trace One-Length Loop", "Model Duplicates",
                    "Model Duplicates Ratio", "Trans No In-arc", "Trans No In-arc ratio", "Silent Transitions",
                    "Silent Transitions ratio", "Parallelism Sum", "Parallelism Sum Ratio", "Parallelism Mult",
                    "Parallelism Mult Ratio", "Choice Sum", "Choice Sum Ratio", "Choice Mult", "Choice Mult Ratio",
                    "Simplicity", "Free-Choice", "Visited States", "Queued States", "Deadlocks", "Boundedness"
                    ]

        features_test = []
        for j in range(n):
            features_test.append(row[features[j]])

        features_test_arr = np.array(features_test)

        features_train_dict = {}
        # features_train_ls = []

        # initialise k-neighbours
        curr_diff = max_difference
        if max_difference > 0:
            iter_intitial_k = 0
            max_iter_initial_k = 0

            while True:
                if max_iter_initial_k > max_iter:
                    break

                len_features_train_dict = len(features_train_dict)
                if iter_intitial_k > 10 * k and len_features_train_dict < k:
                    curr_diff += 0.1 * max_difference
                    # print("Curr difference ", curr_diff)
                    iter_intitial_k = 0

                rand_idx = random.randrange(len(df_train))
                rand_idx_features = []

                row_train = df_train.iloc[rand_idx, :]
                for j in range(n):
                    rand_idx_features.append(row_train[features[j]])

                rand_idx_features_arr = np.array(rand_idx_features)

                # compute distance to test
                curr_dist = np.linalg.norm(features_test_arr - rand_idx_features_arr)  # different norms possible
                if curr_dist <= curr_diff:
                    features_train_dict[rand_idx] = [rand_idx_features_arr, curr_dist]

                iter_intitial_k += 1
                max_iter_initial_k += 1

            # print("Initial curr features train dict ", features_train_dict)

        # find closest features in training
        num_iter = 0
        while True:
            if num_iter >= max_iter:
                break

            # check randomly for k-nearest instances in n features of training set
            # iteratively, randomly?
            rand_idx = random.randrange(len(df_train))
            rand_idx_features = []

            row_train = df_train.iloc[rand_idx, :]
            for j in range(n):
                rand_idx_features.append(row_train[features[j]])

            rand_idx_features_arr = np.array(rand_idx_features)

            # compute distance to test
            curr_dist = np.linalg.norm(features_test_arr - rand_idx_features_arr)  # different norms possible

            # if not k closest yet add, else compare
            if len(features_train_dict) < k:
                features_train_dict[rand_idx] = [rand_idx_features_arr, curr_dist]
            else:
                # find neighbour with highest difference and replace it evtl.
                key_max_dist = max(features_train_dict, key=lambda x: features_train_dict[x][1])
                max_dist = features_train_dict[key_max_dist][1]

                if max_dist > curr_dist:
                    # features_train_dict.pop(key_max_dist)
                    del features_train_dict[key_max_dist]
                    features_train_dict[rand_idx] = [rand_idx_features_arr, curr_dist]

            num_iter += 1

        # get labels and count occurences
        keys_ls = list(features_train_dict.keys())
        count_labels = {"Naive": 0, "No Heuristic": 0, "State Eq. LP": 0, "State Eq. ILP": 0, "Ext. Eq. LP": 0,
                        "Ext. Eq. ILP": 0}

        for i in keys_ls:
            label = y_train["Label"].iloc[i]
            count_labels[label] = count_labels.get(label) + 1

        # majority vote on label?
        # add certain value to probability estimate?
        max_votes_val = max(count_labels.values())
        key_max_votes = max(count_labels, key=lambda x: count_labels[x])

        frequency_max_votes = sum(x == max_votes_val for x in count_labels.values())
        if frequency_max_votes > 1:
            majority_ambiguous += 1

        if weight == "majority":
            prediction = key_max_votes
        else:
            prediction = max(prob_dens, key=lambda x: prob_dens[x])
        recommendation_ls.append(prediction)

    print(features_test_arr)
    print(features_train_dict)
    print(count_labels)

    return recommendation_ls, count_do_k_neighbour, majority_ambiguous


def recommendation_bounding_box(df_train, y_train, df_test, df_proba, max_prob_thres, k, n, max_difference,
                                weight="majority"):
    df_test = df_test.reset_index()
    test_proba = df_test.join(df_proba, lsuffix="_alignment")
    df_train = df_train.reset_index()
    y_train = y_train.reset_index()

    recommendation_ls = []
    majority_ambiguous = 0
    neighbours = []
    count_do_bounding_neighbour = 0
    for i in test_proba.index:

        row = test_proba.iloc[i, :]

        prob_dens = {"No Heuristic": row["No Heuristic"], "Naive": row["Naive"], "State Eq. LP": row["State Eq. LP"],
                     "Ext. Eq. LP": row["Ext. Eq. LP"]}

        max_prob = max(prob_dens.values())

        if max_prob > max_prob_thres:
            prediction = max(prob_dens, key=lambda x: prob_dens[x])
            recommendation_ls.append(prediction)
            continue

        count_do_bounding_neighbour += 1

        # retrieve features of test
        features = ["Matching Labels Trace", "Matching Trace Ratio", "Matching Labels Models", "Matching Model Ratio",
                    "Matching Starts", "Matching Ends", "Trace Length", "Trace Trans Ratio", "Trace Place Ratio",
                    "Trans Trace Ratio", "Place Trace Ratio", "Distinct Events", "Trace Loop", "Trace Loop Ratio",
                    "Trace Loop Max", "Trace Loop Max Ratio", "Trace Loop Mean", "Trace Loop Mean Ratio",
                    "Trace Loop Sum", "Trace Loop Sum Ratio", "Trace One-Length Loop", "Model Duplicates",
                    "Model Duplicates Ratio", "Trans No In-arc", "Trans No In-arc ratio", "Silent Transitions",
                    "Silent Transitions ratio", "Parallelism Sum", "Parallelism Sum Ratio", "Parallelism Mult",
                    "Parallelism Mult Ratio", "Choice Sum", "Choice Sum Ratio", "Choice Mult", "Choice Mult Ratio",
                    "Simplicity", "Free-Choice", "Visited States", "Queued States", "Deadlocks", "Boundedness"]

        features_test = []
        for j in range(n):
            # append features of test row to list
            features_test.append(row[features[j]])

        features_test_arr = np.array(features_test)

        features_train_dict = {}

        curr_diff = max_difference

        # compute distances to all training data
        for i in df_train.index:
            # get features of each row
            row_train = df_train.iloc[i, :]
            row_features = []

            for j in range(n):
                row_features.append(row_train[features[j]])

            row_features_arr = np.array(row_features)

            # compute distance to test
            curr_dist = np.linalg.norm(features_test_arr - row_features_arr)

            features_train_dict[i] = [row_features_arr, curr_dist]

        bounding_box = {}
        # find closest neighbours
        while len(bounding_box) < 1:

            # choose rows with distance < curr_diff
            for i in df_train.index:
                if features_train_dict[i][1] <= curr_diff:
                    bounding_box[i] = [features_train_dict[i][0], features_train_dict[i][1]]

            # increase curr_diff
            curr_diff += 0.1 * max_difference

        # get labels and count occurences
        keys_ls = list(bounding_box.keys())
        count_labels = {"Naive": 0, "No Heuristic": 0, "State Eq. LP": 0, "State Eq. ILP": 0, "Ext. Eq. LP": 0,
                        "Ext. Eq. ILP": 0}

        for i in keys_ls:
            label = y_train["Label"].iloc[i]
            count_labels[label] = count_labels.get(label) + 1

        # majority vote on label?
        # add certain value to probability estimate?
        key_max_votes = max(count_labels, key=lambda x: count_labels[x])
        max_votes_val = max(count_labels.values())

        frequency_max_votes = sum(x == max_votes_val for x in count_labels.values())
        if frequency_max_votes > 1:
            majority_ambiguous += 1

        if weight == "majority":
            prediction = key_max_votes
        else:
            prediction = max(prob_dens, key=lambda x: prob_dens[x])
        recommendation_ls.append(prediction)
        neighbours.append(len(bounding_box))
        # print("Neighbours in bounding box", len(bounding_box))

    print(count_labels)
    print(neighbours)

    return recommendation_ls, count_do_bounding_neighbour, majority_ambiguous


def evaluate_recommendation_function(X_train, y_train, X_test, y_test, proba_df, num_iterations, k_neighbours,
                                     n_features, prob_threshold, fixed_k, model_name):
    start_recommend = time.time()

    if fixed_k:
        y_pred_recommendation_function, count_do_k_neighbour, majority_ambiguous = recommendation_function(X_train,
                                                                                                           y_train,
                                                                                                           X_test,
                                                                                                           proba_df,
                                                                                                           num_iterations,
                                                                                                           k_neighbours,
                                                                                                           n_features,
                                                                                                           max_difference=0.025 * n_features,
                                                                                                           prob_threshold=prob_threshold)
        print("Times k-nearest neighbour executed", count_do_k_neighbour)

    else:
        y_pred_recommendation_function, count_do_k_neighbour, majority_ambiguous = recommendation_bounding_box(X_train,
                                                                                                               y_train,
                                                                                                               X_test,
                                                                                                               proba_df,
                                                                                                               prob_threshold,
                                                                                                               k_neighbours,
                                                                                                               n_features,
                                                                                                               max_difference=0.02 *
                                                                                                                              n_features)
        print("Times bounding box executed", count_do_k_neighbour)

    end_recommend = time.time()
    print("Times majority ambiguous", majority_ambiguous)
    time_recommend = end_recommend - start_recommend
    print("Recommendation function parameters: " + str(num_iterations) + " iterations " +
          str(k_neighbours) + " neighbours" + str(n_features) + " features")
    print(model_name, "+ recommendation function: Number of mislabeled points out of a total %d points : %d" % (
        X_test.shape[0], (np.asarray(y_test_ls) != np.asarray(y_pred_recommendation_function)).sum()))

    score = metrics.accuracy_score(y_test, y_pred_recommendation_function)
    print(model_name, "+ recommendation function accuracy for :   %0.3f" % score)

    class_report = metrics.classification_report(y_test, y_pred_recommendation_function, target_names=mlp_labels)

    cm = metrics.confusion_matrix(y_test, y_pred_recommendation_function, labels=mlp_labels)

    df_cm = pd.DataFrame(cm, columns=mlp_labels, index=mlp_labels)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    # plt.figure(figsize=(20, 7))
    # sns.set(font_scale=1.6)  # for label size
    # sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')
    # plt.title(
    #   "Confusions Matrix of Neural Network + Recommendation Function for " + str(num_iterations) + " iterations " +
    #  str(k_neighbours) + " neighbours" + str(n_features) + " features")
    # sns.heatmap(cm, annot=mlp_labels, cmap='Blues')
    # plt.show()
    return y_pred_recommendation_function, time_recommend


def compare_recommendations(y_true, y_pred1, y_pred2):
    comp_true_pred1 = list(i[0] == i[1] for i in zip(y_true, y_pred1))
    comp_true_pred2 = list(i[0] == i[1] for i in zip(y_true, y_pred2))

    count_better = 0
    count_worse = 0
    count_same = 0

    count_state_no_better = 0
    count_state_naive_better = 0
    count_state_ext_better = 0
    count_state_no_worse = 0
    count_state_naive_worse = 0
    count_state_ext_worse = 0

    count_no_state_better = 0
    count_no_naive_better = 0
    count_no_ext_better = 0
    count_no_state_worse = 0
    count_no_naive_worse = 0
    count_no_ext_worse = 0

    count_naive_state_better = 0
    count_naive_no_better = 0
    count_naive_ext_better = 0
    count_naive_state_worse = 0
    count_naive_no_worse = 0
    count_naive_ext_worse = 0

    count_ext_state_better = 0
    count_ext_no_better = 0
    count_ext_naive_better = 0
    count_ext_state_worse = 0
    count_ext_no_worse = 0
    count_ext_naive_worse = 0

    for i in range(len(y_true)):
        if comp_true_pred1[i] and comp_true_pred2[i] or not comp_true_pred1[i] and not comp_true_pred2[i]:
            count_same += 1

        if comp_true_pred1[i] and not comp_true_pred2[i]:
            count_worse += 1

        if not comp_true_pred1[i] and comp_true_pred2[i]:
            count_better += 1

    return count_same, count_better, count_worse


def distribution_of_recommendation(y_pred):
    count_no = 0
    count_naive = 0
    count_state_lp = 0
    count_state_ilp = 0
    count_ext_lp = 0
    count_ext_ilp = 0

    for i in y_pred:
        if i == "No Heuristic":
            count_no += 1
        elif i == "Naive":
            count_naive += 1
        elif i == "State Eq. LP":
            count_state_lp += 1
        elif i == "State Eq. ILP":
            count_state_ilp += 1
        elif i == "Ext. Eq. LP":
            count_ext_lp += 1
        elif i == "Ext. Eq. ILP":
            count_ext_ilp += 1

    return count_no, count_naive, count_state_lp, count_state_ilp, count_ext_lp, count_ext_ilp


def get_times_of_each_label(all_data, true_label, pred_label, label):
    # df_y_test = y_test.to_frame()
    # df_y_test = df_y_test.reset_index()
    # df_y_pred = pd.DataFrame(y_pred_mlp, columns=["Prediction"])
    # joined_df = df

    count_true = 0
    count_false = 0

    time_no = 0
    time_naive = 0
    time_ext = 0
    time_state = 0

    time_pred = 0

    # join true with pred label
    joined_df = true_label.join(pred_label)

    # re-set index
    # joined_df = joined_df.set_index('index')

    # filter
    label_filter = joined_df[joined_df['Label'] == label].reset_index(drop=True)

    for i in label_filter.index:
        row = label_filter.iloc[i, :]
        prediction = row['Prediction']
        index_original = row['index']

        if prediction == label:
            count_true += 1
        else:
            count_false += 1

        row_all_data = all_data.iloc[index_original, :]
        if prediction == "State Eq." or prediction == "Ext. Eq.":
            time_pred += row_all_data[prediction + ' LP Time']
        else:
            time_pred += row_all_data[prediction + ' Time']

        time_no += row_all_data['No Heuristic Time']
        time_naive += row_all_data['Naive Time']
        time_state += min(row_all_data['State Eq. LP Time'], row_all_data['State Eq. ILP Time'])
        time_ext += min(row_all_data['Ext. Eq. LP Time'], row_all_data['Ext. Eq. ILP Time'])

    if count_true + count_false != 0:
        acc = count_true / (count_true + count_false)
    else:
        acc = 0

    return count_true, count_false, acc, time_pred, time_no, time_naive, time_state, time_ext


def evaluate_each_label(all_data, true_label, pred_label, use_ilp=False):
    if use_ilp:
        heuristics = ["No Heuristic", "Naive", "State Eq. LP", "State Eq. ILP", "Ext. Eq. LP",
                      "Ext. Eq. ILP"]
    else:
        heuristics = ["No Heuristic", "Naive", "State Eq. LP", "Ext. Eq. LP"]

    for h in heuristics:
        print("Class label " + h)
        num_true, num_false, acc, t_pred, t_no, t_naive, t_state, t_ext = get_times_of_each_label(all_data, true_label,
                                                                                                  pred_label, h)
        print("No. true predictions ", num_true)
        print("No. false predictions ", num_false)
        print("Accuracy ", acc)
        print("Time for computing alignments with prediction ", t_pred)
        print("Time for computing alignments with no heuristic ", t_no)
        print("Time for computing alignments with naive heuristic ", t_naive)
        print("Time for computing alignments with state eq. ", t_state)
        print("Time for computing alignments with ext. eq. ", t_ext)


def compute_prediction_highest_prob_mulit(proba_rf, classes, multi_label_clf):
    predictions = []
    if multi_label_clf == "MLP multi-label":
        len_proba = len(proba_rf)
    else:
        len_proba = len(proba_rf[0])

    for i in range(len_proba):
        if multi_label_clf == "MLP multi-label":
            prob_classes = proba_rf[i].tolist()
        else:
            prob_classes = []
            for j in range(len(classes)):
                prob_classes.append(proba_rf[j][i][1])

        index_max_prob = prob_classes.index(max(prob_classes))
        predictions.append(classes[index_max_prob])

    return predictions


def compute_probabilities_multi_label(proba_rf, classes, multi_label_clf):
    probs = []

    if multi_label_clf == "MLP multi-label":
        len_proba = len(proba_rf)
    else:
        len_proba = len(proba_rf[0])

    for i in range(len_proba):
        if multi_label_clf == "MLP multi-label":
            prob_classifiers = proba_rf[i].tolist()
        else:
            prob_classifiers = []
            for j in range(len(classes)):
                prob_classifiers.append(proba_rf[j][i][1])
        probs.append(prob_classifiers)

    probs_df = pd.DataFrame(probs, columns=classes.tolist())
    return probs_df


def recommend():
    pass


if __name__ == "__main__":

    data = get_merged_data()
    plot_data.create_sunburst_plot(data, 180, "overall")
    data_labeled = label_data(data)
    data_sampled = sample_data(data_labeled, drop_frac=0.95)
    data_ranked = rank_heuristics(data_sampled, use_ilp=False)
    data_features, len_features, time_compute_feature = get_features_data(data_ranked)

    # print(resource.getrlimit(resource.RLIMIT_STACK))
    # print(sys.getrecursionlimit())

    # max_rec = 0x100000

    # May segfault without this line. 0x100 is a guess at the size of each stack frame.
    # resource.setrlimit(resource.RLIMIT_STACK, [0x100 * max_rec, resource.RLIM_INFINITY])
    # sys.setrecursionlimit(max_rec)

    data_features.to_pickle("data_features_updated_eqs.pkl")
    # data_features.to_pickle("undersampeled_0,95_data_features_updated_eqs_0719.pkl")
    #data_features = pd.read_pickle("undersampeled_0,95_data_features_updated_eqs_0719.pkl")

    # data_features = pd.read_pickle("data_features0619.pkl")
    len_features = 41

    data_multi_labeled = multi_label_data(data_features)

    X_train_raw, y_train_raw, X_test_raw, y_test_raw = get_train_test_set(data_sampled)
    data_features_X, len_features_X, time_compute_feature_X = get_features_data(X_test_raw)

    X_train, y_train, X_test, y_test = get_train_test_set(data_features)
    # drop_i = X_train[X_train['Ext. LP Rank'] == 1].index
    # X_train = X_train.drop(drop_i)
    # y_train = y_train.drop(drop_i)

    # drop_i = X_test[X_test['Ext. LP Rank'] == 1].index
    # X_test = X_test.drop(drop_i)
    # y_test = y_test.drop(drop_i)
    timeout = 180
    #plot_data.create_sunburst_plot(X_train, timeout)
    #plot_data.create_sunburst_plot(X_test, timeout)

    # road_x_train = np.asarray(road_x_train.iloc[:, -len_features:], dtype=object)
    X_test_features = np.asarray(X_test.iloc[:, -len_features:], dtype=object)
    X_train_features = np.asarray(X_train.iloc[:, -len_features:], dtype=object)

    # multi label
    X_train_multi, y_train_multi, X_test_multi, y_test_multi = get_train_test_set(data_multi_labeled, multi_label=True)
    X_test_features_multi = np.asarray(X_test_multi.iloc[:, -len_features:], dtype=object)
    X_train_features_multi = np.asarray(X_train_multi.iloc[:, -len_features:], dtype=object)

    mlb = MultiLabelBinarizer()
    y_train_multi_label = mlb.fit_transform(y_train_multi['Label Multi'].values.tolist())
    y_test_multi_label = mlb.fit_transform(y_test_multi['Label Multi'].values.tolist())

    multi_classes = mlb.classes_

    #multi_label_classifier = "RF multi-label"
    multi_label_classifier = "MLP multi-label"
    if multi_label_classifier == "MLP multi-label":
        clf = MLPClassifier(solver='adam', hidden_layer_sizes=(len_features, len_features, 10, 4),
                            learning_rate_init=0.001,
                            random_state=1,
                            max_iter=1000)
    else:
        clf = RandomForestClassifier(max_depth=6, random_state=0)

    clf.fit(X_train_features_multi, y_train_multi_label)

    start_multi_label_predict = time.time()
    proba_rf = clf.predict_proba(X_test_features_multi)
    time_multi_label_predict = time.time() - start_multi_label_predict
    print("Time to predict with", multi_label_classifier, time_multi_label_predict)
    # y_pred_rf = clf.predict(X_test_features_multi)
    # y_pred_rf_inversed = mlb.inverse_transform(y_pred_rf)

    # y_pred_single_label = [y[0] for y in y_pred_rf_inversed]
    # y_pred_single_label_last = [y[-1] if len(y) > 1 else y[0] for y in y_pred_rf_inversed]
    y_pred_single_label = compute_prediction_highest_prob_mulit(proba_rf, multi_classes, multi_label_classifier)

    # oversample = RandomOverSampler(sampling_strategy='not majority')
    # X_over, y_over = oversample.fit_resample(X_train_features, y_train)
    # print("Oversampled class distribution:", Counter(y_over))
    # undersample = RandomUnderSampler(sampling_strategy="majority", random_state=0)
    # X_under, y_under = undersample.fit_resample(X_train_features, y_train)
    # print("Undersampled class distribution:", Counter(y_under))
    # X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=0)

    # NN
    #multi_classes_classifier = "RF multi-class"
    multi_classes_classifier = "MLP multi-class"
    if multi_classes_classifier == "MLP multi-class":
        # start_train_mlp = time.time()
        mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(len_features, len_features, 10, 4),
                            learning_rate_init=0.001,
                            random_state=1,
                            max_iter=1000)
        mlp.out_activation_ = 'softmax'
    else:
        mlp = RandomForestClassifier(max_depth=6, random_state=0)

    mlp.fit(X_train_features, y_train)
    # mlp.fit(X_under, y_under)

    # end_train_mlp = time.time()
    # time_train_mlp = end_train_mlp - start_train_mlp

    # print("Time to compute features: ", time_features)
    # print("Time to train mlp: ", time_train_mlp)

    start_predict_mlp = time.time()
    # y_pred_mlp = mlp.predict(X_test_features)
    y_pred_mlp = mlp.predict(X_test_features)

    end_predict_mlp = time.time()
    time_predict_mlp = end_predict_mlp - start_predict_mlp
    print("Time to predict with:", multi_classes_classifier, time_predict_mlp)

    mlp_labels = mlp.classes_

    y_test_np = y_test.to_numpy()
    y_test_ls = [element for innerList in y_test_np for element in innerList]
    print(multi_classes_classifier, ": Number of mislabeled points out of a total %d points : %d" % (
        X_test.shape[0], (y_test_ls != y_pred_mlp).sum()))

    score = metrics.accuracy_score(y_test, y_pred_mlp)
    print(multi_classes_classifier, "accuracy:   %0.3f" % score)

    class_report = metrics.classification_report(y_test, y_pred_mlp, target_names=mlp_labels)

    cm = metrics.confusion_matrix(y_test, y_pred_mlp, labels=mlp_labels)

    df_cm = pd.DataFrame(cm, columns=mlp_labels, index=mlp_labels)
    df_cm.index.name = 'Actual'
    df_cm.columns.name = 'Predicted'
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.6)  # for label size
    sns.heatmap(df_cm, cmap="Blues", annot=True, annot_kws={"size": 16}, fmt='g')
    plt.title("Confusions Matrix of Neural Network")
    # sns.heatmap(cm, annot=mlp_labels, cmap='Blues')
    plt.show()
    # disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp_labels)
    # disp.plot()
    # plt.show()

    df_y_test = y_test.copy()
    df_y_test = df_y_test.reset_index()
    df_y_pred_rf = pd.DataFrame(y_pred_single_label, columns=["Prediction"])
    # df_y_pred_rf_last = pd.DataFrame(y_pred_single_label_last, columns=["Prediction"])
    df_y_pred_mlp = pd.DataFrame(y_pred_mlp, columns=["Prediction"])

    print("evaluate each label for", multi_label_classifier)
    evaluate_each_label(data_features, df_y_test, df_y_pred_rf)
    print("evaluate each label for", multi_classes_classifier)
    evaluate_each_label(data_features, df_y_test, df_y_pred_mlp)

    rf_idx, rf_time, rf_timeouts, rf_lps, rf_queued = statistical_numbers.return_model_metrics(X_test,
                                                                                               y_pred_single_label,
                                                                                               timeout)
    # rf_idx_last, rf_time_last, rf_timeouts_last, rf_lps_last, rf_queued_last = statistical_numbers.return_model_metrics(
    #   X_test, y_pred_single_label_last,
    #  timeout)

    y_pred_mlp_ls = y_pred_mlp.tolist()
    mlp_idx, mlp_time, mlp_timeouts, mlp_lps, mlp_queued = statistical_numbers.return_model_metrics(X_test,
                                                                                                    y_pred_mlp_ls,
                                                                                                    timeout)
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rcParams["figure.figsize"] = (15, 10)
    statistical_numbers.evaluate_time_only_heuristics(X_test, True)

    time_compute_feature_X = 0
    statistical_numbers.evaluate_time(X_test, [rf_time, mlp_time], [multi_label_classifier, multi_classes_classifier],
                                     time_compute_feature_X, use_features=True, use_ilp=False, plot=True)

    statistical_numbers.evaluate_lps(X_test, [rf_lps, mlp_lps], [multi_label_classifier, multi_classes_classifier],
                                     use_ilp=False, plot=True)

    statistical_numbers.evaluate_queued(X_test, [rf_queued, mlp_queued],
                                       [multi_label_classifier, multi_classes_classifier], use_ilp=False, plot=True)

    statistical_numbers.evaluate_timeouts(X_test, [rf_timeouts, mlp_timeouts], [multi_label_classifier, multi_classes_classifier], timeout)
    print("Time with", multi_classes_classifier, mlp_time)
    print("Time with", multi_label_classifier, rf_time)

    # k = 3
    # for i in range(k):
    #   rank_in_top_k = ranking_accuracy_top_k(X_test, proba_ranks, i)
    #  print("Rank accuracy of top ", k, ": ", rank_in_top_k)

    # Ensembles

    y_test_np = y_test.to_numpy()
    y_pred_relax_lp = y_pred_mlp.copy()

    count_state = 0
    for i in range(len(y_test)):
        if y_test_np[i] == "State Eq. LP" and y_pred_mlp[i] == "State Eq. ILP":
            y_pred_relax_lp[i] = "State Eq. LP"
            count_state += 1
        elif y_test_np[i] == "State Eq. ILP" and y_pred_mlp[i] == "State Eq. LP":
            y_pred_relax_lp[i] = "State Eq. ILP"
            count_state += 1
    print("LP version predicted, ILP version is truth and vice versa: ", count_state)
    score_relax = metrics.accuracy_score(y_test, y_pred_relax_lp)
    print(multi_classes_classifier, "accuracy when no difference between LP and ILP:   %0.3f" % score_relax)

    time_relaxed_model = statistical_numbers.time_using_model(X_test, y_pred_relax_lp)

    """
    # random model
    # labels = ["Naive", "No Heuristic", "State Eq. LP", "State Eq. ILP",
    # "Ext. Eq. LP", "Ext. Eq. ILP"
    #         ]
    len_x_test = len(X_test)
    random_predictions = []
    for i in mlp_labels:
        random_predictions = random_predictions + [i for j in range(math.floor(len_x_test / len(mlp_labels)))]

    while len(random_predictions) < len_x_test:
        for i in mlp_labels:
            random_predictions.append(i)

            if len(random_predictions) == len_x_test:
                break
    random.seed(0)
    random.shuffle(random_predictions)

    random_time = statistical_numbers.time_using_model(X_test, random_predictions)
    random_idx = statistical_numbers.replace_label_by_time(random_predictions)
    random_timeouts, _ = statistical_numbers.timeouts_optimal_heuristics(X_test, 180, random_idx)
    random_lps = statistical_numbers.lps_optimal_heuristics(X_test, random_idx)
    random_visited_states = statistical_numbers.states_optimal_heuristics(X_test, True, random_idx)
    random_queued_states = statistical_numbers.states_optimal_heuristics(X_test, False, random_idx)

    statistical_numbers.plot_multiple_bars_h(optimal_time, [random_time, mlp_time], time_heuristics,
                                             "Time in seconds", "Computation Time MLP Model")
    statistical_numbers.plot_multiple_bars_h(optimal_queued_states, [random_queued_states, mlp_queued],
                                             queued_states_heuristic, "Number of queued states",
                                             "Queued States")
    """
    """
    statistical_numbers.plot_multiple_bars(optimal_time, [random_time, time_relaxed_model], time_heuristics,
                                           "Time in seconds", "Computation Time Relaxed Model")
    
    statistical_numbers.plot_multiple_bars(optimal_timeouts, [random_timeouts, model_timeouts], timeouts_heuristics,
                                           "Number of timeouts", plot_title="Timeouts")
    statistical_numbers.plot_multiple_bars(optimal_lps, [random_lps, model_lps], lps_one_heuristic,
                                           "Number of solved lps", "Solved LPs")
    statistical_numbers.plot_multiple_bars(optimal_visited_states, [random_visited_states, model_visited_states],
                                           visited_states_heuristic,
                                           "Number of visited states", "Visited States")
    """
    """
    k nearest neighbour
    """

    proba_mlp = mlp.predict_proba(X_test_features)
    proba_df = pd.DataFrame(proba_mlp, columns=mlp_labels)
    proba_ranks = rank_heuristics(proba_df, use_ilp=False, reverse=True, column_no="No Heuristic", column_naive="Naive",
                                  column_state_lp="State Eq. LP", column_ext_lp="Ext. Eq. LP")

    proba_rf_df = compute_probabilities_multi_label(proba_rf, multi_classes, multi_label_classifier)
    # num_iterations = [250, 500, 1000, 1500]
    # num_iterations = [1000, 1500]
    len_X_test = len(X_test)
    #num_iterations = [0.5 * len_X_test, 0.75 * len_X_test, len_X_test]
    num_iterations = [len_X_test]
    k_neighbours = [5, 10, 15]
    # n_features = [math.floor(len_features/4), math.floor(len_features/2), math.floor(len_features * 0.75), len_features]
    n_features = len_features
    fixed_k = True
    probs_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

    recommend_mlp = True
    # k nearest neighbour
    for fixed_k in [True]:
        for prob_thres in probs_thresholds:
            if fixed_k:
                #y_recommendations = [[0 for k in range(len(k_neighbours))] for i in range(len(num_iterations))]
                for j in range(len(num_iterations)):
                    for k in range(len(k_neighbours)):
                        # for n in n_features:
                        print(prob_thres, num_iterations[j], k_neighbours[k], n_features)
                        for i in range(2):
                            if recommend_mlp:
                                print(multi_classes_classifier, "+ recommendation")
                                y_recommend, time_recommend = evaluate_recommendation_function(X_train, y_train, X_test, y_test, proba_df,
                                                                                               num_iterations[j], k_neighbours[k],
                                                                                               n_features, prob_threshold=prob_thres,
                                                                                               fixed_k=True, model_name=multi_classes_classifier)

                            else:
                                print(multi_label_classifier, "+ recommendation")
                                y_recommend, time_recommend = evaluate_recommendation_function(X_train, y_train, X_test, y_test,
                                                                                               proba_rf_df,
                                                                                               num_iterations[j], k_neighbours[k],
                                                                                               n_features, prob_threshold=prob_thres,
                                                                                               fixed_k=True, model_name=multi_label_classifier)

                            #y_recommendations[i][k] = [y_recommend, time_recommend]
                            print("Time to compute recommendation ", time_recommend)

                            # recommendation_function(X_train, y_train, X_test, y_test, proba_df, i, k, n_features)

                            # show statistics
                            #y_pred_mlp_ls = y_recommendations[i][k][0]
                            y_pred_mlp_ls = y_recommend
                            #recommend_time = y_recommendations[i][k][1]
                            recommend_time = time_recommend

                            # model_idx = statistical_numbers.replace_label_by_time(y_pred_mlp_ls)
                            model_recommend_time = statistical_numbers.time_using_model(X_test, y_pred_mlp_ls)
                            print("Model recommend time: ", model_recommend_time)

                            model_recommend_idx, model_time, model_recommend_timeouts, model_recommend_lps, model_recommend_queued = statistical_numbers.return_model_metrics(X_test,
                                                                                                                       y_pred_mlp_ls,
                                                                                                                       timeout)
                            if recommend_mlp:
                                model_name = "MLP"

                                count_same, count_better, count_worse = compare_recommendations(y_test_ls, y_pred_mlp,
                                                                                                y_pred_mlp_ls)
                                print("Recommendation function compared to mlp ", count_same, count_better, count_worse)

                                recommend_mlp = False
                            else:
                                model_name = "RF"
                                count_same, count_better, count_worse = compare_recommendations(y_test_ls,
                                                                                                y_pred_single_label,
                                                                                                y_recommend)
                                print("Recommendation function compared to rf ", count_same, count_better, count_worse)
                                recommend_mlp = True


                            statistical_numbers.evaluate_time(X_test, [model_recommend_time, rf_time, mlp_time],
                                                              [model_name+ str(num_iterations[j]) + " iterations and"
                                                                           + str(k_neighbours[k]) + " neighbours", "RF", "MLP"], plot=False)
                            statistical_numbers.evaluate_lps(X_test, [model_recommend_lps, rf_lps, mlp_lps], [model_name+ str(num_iterations[j]) + " iterations and"
                                                                           + str(k_neighbours[k]) + " neighbours","RF", "MLP"], plot=False)

                            statistical_numbers.evaluate_queued(X_test, [model_recommend_queued, rf_queued, mlp_queued],
                                                                [model_name+ str(num_iterations[j]) + " iterations and"
                                                                           + str(k_neighbours[k]) + " neighbours", "RF", "MLP"], plot=False)

                            statistical_numbers.evaluate_timeouts(X_test, [model_recommend_timeouts, rf_timeouts, mlp_timeouts],
                                                                  [model_name+ str(num_iterations[j]) + " iterations and"
                                                                           + str(k_neighbours[k]) + " neighbours", "RF", "MLP"], timeout, plot=False)



            else:
                print("bounding box with probability threshold", prob_thres)

                for i in range(2):
                    if recommend_mlp:
                        print(multi_classes_classifier,"+ recommendation")
                        y_recommend, time_recommend = evaluate_recommendation_function(X_train, y_train, X_test, y_test, proba_df,
                                                                               X_test.shape[0], k_neighbours[0],
                                                                               n_features, prob_thres, fixed_k=False, model_name=multi_classes_classifier)
                    else:
                        print(multi_label_classifier, "+ recommendation")
                        y_recommend, time_recommend = evaluate_recommendation_function(X_train, y_train, X_test, y_test,
                                                                                       proba_rf_df,
                                                                                       X_test.shape[0], k_neighbours[0],
                                                                                       n_features, prob_thres,
                                                                                       fixed_k=False, model_name=multi_label_classifier)

                    print("Time to compute recommendation ", time_recommend)

                    model_recommend_time = statistical_numbers.time_using_model(X_test, y_recommend)
                    print("Model recommend time: ", model_recommend_time)

                    model_recommend_idx, model_time, model_recommend_timeouts, model_recommend_lps, model_recommend_queued = statistical_numbers.return_model_metrics(
                        X_test,
                        y_pred_mlp_ls,
                        timeout)

                    if recommend_mlp:
                        model_name = "MLP"
                        count_same, count_better, count_worse = compare_recommendations(y_test_ls, y_pred_mlp, y_recommend)
                        print("Recommendation function compared to mlp ", count_same, count_better, count_worse)

                        recommend_mlp = False
                    else:
                        model_name = "RF"
                        count_same, count_better, count_worse = compare_recommendations(y_test_ls, y_pred_single_label, y_recommend)
                        print("Recommendation function compared to rf ", count_same, count_better, count_worse)
                        recommend_mlp = True

                    statistical_numbers.evaluate_time(X_test, [model_recommend_time, rf_time, mlp_time],
                                                      [model_name + " bounding box", "RF", "MLP"], plot=False)
                    statistical_numbers.evaluate_lps(X_test, [model_recommend_lps, rf_lps, mlp_lps],
                                                     [model_name + " bounding box", "RF", "MLP"], plot=False)

                    statistical_numbers.evaluate_queued(X_test, [model_recommend_queued, rf_queued, mlp_queued],
                                                        [model_name + " bounding box", "RF", "MLP"], plot=False)

                    statistical_numbers.evaluate_timeouts(X_test, [model_recommend_timeouts, rf_timeouts, mlp_timeouts],
                                                          [model_name + " bounding box", "RF", "MLP"], timeout, plot=False)

