import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import math


def time_with_optimal_heuristics(df):
    times = df[["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time", "Ext. Eq. LP Time",
                "Ext. Eq. ILP Time"]].min(axis=1)

    print("Time with optimal heuristics: " + str(times.sum()))
    return round(times.sum(), 2)


def time_using_one_heuristic(df):
    heuristics = ["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time", "Ext. Eq. LP Time",
                  "Ext. Eq. ILP Time"]

    times = []

    for h in heuristics:
        times.append(round(df[h].sum(), 2))

    return times


def lps_using_one_heuristic(df):
    h_lps = ["State Eq. LP Solved LP", "State Eq. ILP Solved LP", "Ext. Eq. LP Solved LP", "Ext. Eq. ILP Solved LP"]

    # filter out timeouts
    list_drop = ["Timeout", "Result None"]
    lps = [0, 0]

    for h in h_lps:
        df = df[df[h].isin(list_drop) == False]

    for h in h_lps:
        lps.append(df[h].sum())

    return lps


def lps_optimal_heuristics(df, index=None):
    if index is None:
        min_index = df[["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time",
                        "Ext. Eq. LP Time", "Ext. Eq. ILP Time"]].idxmin(axis="columns")
    else:
        min_index = index
    num_lps = 0
    row_num = 0

    for i in min_index:
        row = df.iloc[row_num, :]

        if i == "State Eq. LP Time":
            num_lps = num_lps + row["State Eq. LP Solved LP"]
        elif i == "State Eq. ILP Time":
            num_lps = num_lps + row["State Eq. ILP Solved LP"]
        elif i == "Ext. Eq. LP Time":
            num_lps = num_lps + row["Ext. Eq. LP Solved LP"]
        elif i == "Ext. Eq. ILP Time":
            num_lps = num_lps + row["Ext. Eq. ILP Solved LP"]

        row_num = row_num + 1

    return num_lps


def replace_label_by_time(ls):
    for i in range(len(ls)):
        if ls[i] == "No Heuristic":
            ls[i] = "No Heuristic Time"
        elif ls[i] == "Naive":
            ls[i] = "Naive Time"
        elif ls[i] == "State Eq. LP":
            ls[i] = "State Eq. LP Time"
        elif ls[i] == "State Eq. ILP":
            ls[i] = "State Eq. ILP Time"
        elif ls[i] == "Ext. Eq. LP":
            ls[i] = "Ext. Eq. LP Time"
        elif ls[i] == "Ext. Eq. ILP":
            ls[i] = "Ext. Eq. ILP Time"
    return ls


def timeouts_optimal_heuristics(df, timeout, index=None):
    if index is None:
        min_index = df[["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time",
                        "Ext. Eq. LP Time", "Ext. Eq. ILP Time"]].idxmin(axis="columns")
    else:
        min_index = index

    num_timeouts = 0
    row_num = 0

    for i in min_index:
        row = df.iloc[row_num, :]
        min_time = row[i]

        if min_time >= timeout:
            num_timeouts += 1

        row_num += 1

    return num_timeouts, num_timeouts / row_num


def states_optimal_heuristics(df, visited, index=None):
    if index is None:
        min_index = df[["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time",
                        "Ext. Eq. LP Time", "Ext. Eq. ILP Time"]].idxmin(axis="columns")
    else:
        min_index = index

    num_states = 0
    row_num = 0

    for i in min_index:
        row = df.iloc[row_num, :]

        if i == "No Heuristic Time":
            opt_alignment = row["No Heuristic"]
        elif i == "Naive Time":
            opt_alignment = row["Naive"]
        elif i == "State Eq. LP Time":
            opt_alignment = row["State Eq. LP"]
        elif i == "State Eq. ILP Time":
            opt_alignment = row["State Eq. ILP"]
        elif i == "Ext. Eq. LP Time":
            opt_alignment = row["Ext. Eq. LP"]
        elif i == "Ext. Eq. ILP Time":
            opt_alignment = row["Ext. Eq. ILP"]

        # visited = True: visited states, else queued states
        if opt_alignment is not None:
            if visited:
                num_states += opt_alignment["visited_states"]
            else:
                num_states += opt_alignment["queued_states"]

        row_num += 1

    return num_states


def states_one_heuristic(df, visited):
    heuristics = ["No Heuristic", "Naive", "State Eq. LP", "State Eq. ILP", "Ext. Eq. LP",
                  "Ext. Eq. ILP"]

    num_states = [[] for i in range(len(heuristics))]

    for ind in df.index:
        row = df.iloc[ind, :]

        for i in range(len(heuristics)):
            if row[heuristics[i]] is not None:
                if visited:
                    num_states[i].append(row[heuristics[i]]["visited_states"])
                else:
                    num_states[i].append(row[heuristics[i]]["queued_states"])
            else:
                num_states[i].append(0)

    num_states = np.sum(num_states, axis=1)

    return num_states.tolist()


def variance_between_heuristics(df, h1, h2):
    pass


def use_one_lp_version(df, use_state_eq, use_relax, index=None):
    if index is None:
        min_index = df[["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time",
                        "Ext. Eq. LP Time", "Ext. Eq. ILP Time"]].idxmin(axis="columns")
    else:
        min_index = index

    time = 0
    row_num = 0

    for i in min_index:
        row = df.iloc[row_num, :]

        if i == "No Heuristic Time":
            time += row["No Heuristic Time"]
        elif i == "Naive Time":
            time += row["Naive Time"]
        else:
            # replace state eq. times
            if use_state_eq:

                if use_relax:
                    if i == "State Eq. LP Time":
                        time += row["State Eq. LP Time"]
                    # replace ilp with lp
                    if i == "State Eq. ILP Time":
                        time += row["State Eq. LP Time"]
                # replace lp with ilp
                else:
                    if i == "State Eq. LP Time":
                        time += row["State Eq. ILP Time"]
                    # replace ilp with lp
                    if i == "State Eq. ILP Time":
                        time += row["State Eq. ILP Time"]
                # do not change anything for ext. eq.
                if i == "Ext. Eq. LP Time":
                    time += row["Ext. Eq. LP Time"]
                if i == "Ext. Eq. ILP Time":
                    time += row["Ext. Eq. ILP Time"]
            else:  # replace ext. eq. times
                if i == "State Eq. LP Time":
                    time += row["State Eq. LP Time"]
                if i == "State Eq. ILP Time":
                    time += row["State Eq. ILP Time"]

                if use_relax:
                    if i == "Ext Eq. LP Time":
                        time += row["Ext. Eq. LP Time"]
                    # replace ilp with lp
                    if i == "Ext. Eq. ILP Time":
                        time += row["Ext. Eq. LP Time"]
                # replace lp with ilp
                else:
                    if i == "Ext. Eq. LP Time":
                        time += row["Ext. Eq. ILP Time"]
                    # replace ilp with lp
                    if i == "Ext. Eq. ILP Time":
                        time += row["Ext. Eq. ILP Time"]

        row_num += 1

    return time


def time_using_model(df, predictions):
    time = 0
    row_num = 0

    for i in predictions:
        row = df.iloc[row_num, :]

        if i == "No Heuristic":
            time += row["No Heuristic Time"]
        elif i == "Naive":
            time += row["Naive Time"]
        elif i == "State Eq. LP":
            time += row["State Eq. LP Time"]
        elif i == "State Eq. ILP":
            time += row["State Eq. ILP Time"]
        elif i == "Ext. Eq. LP":
            time += row["Ext. Eq. LP Time"]
        elif i == "Ext. Eq. ILP":
            time += row["Ext. Eq. ILP Time"]

        row_num += 1

    return round(time, 2)


def plot_multiple_bars(optimal, model, ls_heuristics, y_label, plot_title):
    x = ["Optimal", "Random", "Dijkstra", "Naive", "State LP", "State ILP", "Ext. LP", "Ext. ILP"]

    #y_optimal = [optimal for i in range(len_x)]

    #y_model = [model for i in range(len_x)]
    z_others = ls_heuristics

    values = []
    values.append(optimal)

    if isinstance(model, list):
        for i in model:
            values.append(i)
        x = ["Optimal", "Random", "Model", "Dijkstra", "Naive", "State LP", "State ILP", "Ext. LP", "Ext. ILP"]
    else:
        values.append(model)
    values = values + ls_heuristics

    #if optimal == 0:
        #absolute difference
        #differences_model = [z - optimal for z in y_model]
        #differences_heuristics = [z - optimal for z in z_others]
    #else:  # percentage difference
        #differences_model = [round(z / optimal, 2) for z in y_model]
        #differences_heuristics = [round(z / optimal, 2) for z in z_others]
    len_x = len(x)
    x_axis = np.arange(len_x)

    bar_width = 0.25

    br1 = x_axis
    # br2 = [x + bar_width for x in br1]
    # br3 = [x + bar_width for x in br2]

    plt.bar(br1, values, bar_width)
    # plt.bar(br2, y_model, bar_width, label="model")
    # plt.bar(br3, z_others, bar_width, label="other heuristics")

    plt.axhline(y=optimal, color="green", linestyle="dashed", label="Optimal")
    if isinstance(model, list):
        plt.axhline(y=model[0], color="r", linestyle="dashed", label="Random")
        plt.axhline(y=model[1], color="b", linestyle="dashed", label="ML")
    else:
        plt.axhline(y=model, color="r", linestyle="dashed", label="Random")

    for i in range(len_x):
        plt.text(i, values[i] + 0.05, round(values[i], 3), ha="center")
    #   plt.text(i + 1 * bar_width, y_model[i], differences_model[i], ha="center")
    #  plt.text(i + 2 * bar_width, z_others[i], differences_heuristics[i], ha="center")

    plt.xticks([r for r in range(len_x)], x)
    plt.xlabel("Comparison of optimal heuristics and only using one heuristic")
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.legend()
    plt.savefig("visualization/" + plot_title + ".png")
    plt.show()


if __name__ == "__main__":

    df = pd.read_pickle("results/road_inductive_0_3.pkl")
    len_df = len(df.index)
    timeout = 180

    optimal_time = time_with_optimal_heuristics(df)
    times_one_heuristic = time_using_one_heuristic(df)

    optimal_lps = lps_optimal_heuristics(df)
    lps_one_heuristic = lps_using_one_heuristic(df)

    time_loss = []

    for t in times_one_heuristic:
        time_loss.append(round(t / optimal_time, 4))

    # timeouts
    optimal_timeouts, _ = timeouts_optimal_heuristics(df, timeout)

    timeouts_heuristics = []

    heuristics = ["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time", "Ext. Eq. LP Time",
                  "Ext. Eq. ILP Time"]

    for h in heuristics:
        num_timeouts = len(df[df[h].ge(timeout)].index)
        timeouts_heuristics.append(num_timeouts)

    # states
    optimal_visited_states = states_optimal_heuristics(df, True)
    optimal_queued_states = states_optimal_heuristics(df, False)

    visited_states_heuristic = states_one_heuristic(df, True)
    queued_states_heuristic = states_one_heuristic(df, False)

    # only one lp version
    times_state_lp = use_one_lp_version(df, use_state_eq=True, use_relax=True)
    times_state_ilp = use_one_lp_version(df, use_state_eq=True, use_relax=False)
    times_ext_lp = use_one_lp_version(df, use_state_eq=False, use_relax=True)
    times_ext_ilp = use_one_lp_version(df, use_state_eq=False, use_relax=False)

    print(times_state_lp)
    print(times_state_ilp)
    print(times_ext_lp)
    print(times_ext_ilp)

    # df.loc['total'] = df[["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time",
    #                     "Ext. Eq. LP Time", "Ext. Eq. ILP Time", "State Eq. LP Solved LP",
    #                    "State Eq. ILP Solved LP", "Ext. Eq. LP Solved LP", "Ext. Eq. ILP Solved LP"]].sum()

    # total_times = df.iloc[-1, 10:16]
    # total_lps = df.iloc[-1, -4:]

    # saved_lps = [round(optimal_lps / x, 4) for x in total_lps]

    print("Total times with each heuristic")
    # print(total_times)
    print(times_one_heuristic)
    print("Time loss ", time_loss)

    print(optimal_lps)
    # print(total_lps)
    # print(saved_lps)

    print("Number of timeouts with optimal heuristics: ", optimal_timeouts)
    print("Timeouts with each heuristic")
    print(timeouts_heuristics)

    print(optimal_visited_states)
    print(optimal_queued_states)
    print("Visited states with each heuristic")
    print(visited_states_heuristic)
    print("Queued states with each heuristic")
    print(queued_states_heuristic)

    # random model
    labels = ["Naive", "No Heuristic", "State Eq. LP", "State Eq. ILP",
              # "Ext. Eq. LP", "Ext. Eq. ILP"
              ]
    random_predictions = []
    for i in labels:
        random_predictions = random_predictions + [i for j in range(math.floor(len_df / len(labels)))]

    while len(random_predictions) < len_df:
        for i in labels:
            random_predictions.append(i)

            if len(random_predictions) == len_df:
                break
    random.seed(0)
    random.shuffle(random_predictions)
    # random_predictions = random.choices(labels, k=len(df.index), weights=[1, 1, 1, 1, 0, 0])
    set_predictions = set(random_predictions)

    random_distribution = []
    for i in set_predictions:
        random_distribution.append(random_predictions.count(i))

    random_time = time_using_model(df, random_predictions)
    random_idx = replace_label_by_time(random_predictions)
    random_timeouts, _ = timeouts_optimal_heuristics(df, timeout, random_idx)
    random_lps = lps_optimal_heuristics(df, random_idx)
    random_visited_states = states_optimal_heuristics(df, True, random_idx)
    random_queued_states = states_optimal_heuristics(df, False, random_idx)

    print("Random guessing: ", random_time)

    plot_multiple_bars(optimal_time, random_time, times_one_heuristic, "Time in seconds", "Computation times")
    plot_multiple_bars(optimal_timeouts, random_timeouts, timeouts_heuristics, "Number of timeouts",
                       plot_title="Timeouts")
    plot_multiple_bars(optimal_lps, random_lps, lps_one_heuristic, "Number of solved lps", "Solved LPs")
    plot_multiple_bars(optimal_visited_states, random_visited_states, visited_states_heuristic,
                       "Number of visited states", "Visited States")
    plot_multiple_bars(optimal_queued_states, random_queued_states, queued_states_heuristic, "Number of queued states",
                       "Queued States")
