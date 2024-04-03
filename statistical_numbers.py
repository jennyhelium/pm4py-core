import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt


def time_with_optimal_heuristics(df):
    times = df[["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time", "Ext. Eq. LP Time",
                "Ext. Eq. ILP Time"]].min(axis=1)

    print("Time with optimal heuristics: " + str(times.sum()))
    return times.sum()


def time_using_one_heuristic(df):
    heuristics = ["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time", "Ext. Eq. LP Time",
                  "Ext. Eq. ILP Time"]

    times = []

    for h in heuristics:
        times.append(df[h].sum())

    return times


def lps_using_one_heuristic(df):
    h_lps = ["State Eq. LP Solved LP", "State Eq. ILP Solved LP", "Ext. Eq. LP Solved LP", "Ext. Eq. ILP Solved LP"]

    lps = [0, 0]

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

    # TODO for index = predictions
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
            if visited:
                num_states[i].append(row[heuristics[i]]["visited_states"])
            else:
                num_states[i].append(row[heuristics[i]]["queued_states"])

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

    return time


def plot_multiple_bars(optimal, ls_heuristics, y_label, plot_title):
    x = ["No Heuristic", "Naive", "State LP", "State ILP", "Ext. LP", "Ext. ILP"]
    len_x = len(x)
    y_optimal = [optimal for i in range(len_x)]
    z_others = ls_heuristics

    if optimal == 0:
        # absolute difference
        differences = [z - optimal for z in z_others]
    else:  # percentage difference
        differences = [round(z / optimal, 2) for z in z_others]

    x_axis = np.arange(len(x))

    plt.bar(x_axis - 0.2, y_optimal, 0.4, label="optimal")
    plt.bar(x_axis + 0.2, z_others, width=0.4, label="other heuristics")

    for i in range(len_x):
        plt.text(i + 0.2, z_others[i], differences[i], ha="center")

    plt.xticks(x_axis, x)
    plt.xlabel("Comparison of optimal heuristics and only using one heuristic")
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.legend()
    plt.show()


if __name__ == "__main__":

    df = pd.read_pickle("results/italian_alpha_0.2_3.pkl")
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
    labels = ["Naive", "No Heuristic", "State Eq. LP", "State Eq. ILP", "Ext. Eq. LP", "Ext. Eq. ILP"]
    random_predictions = random.choices(labels, k=len(df.index), weights=[1, 1, 1, 1, 0, 0])

    print("Random guessing: ", time_using_model(df, random_predictions))

    plot_multiple_bars(optimal_time, times_one_heuristic, "Time in seconds", "Computation times")
    plot_multiple_bars(optimal_timeouts, timeouts_heuristics, "Number of timeouts", plot_title="Timeouts")
    plot_multiple_bars(optimal_lps, lps_one_heuristic, "Number of solved lps", "Solved LPs")
    plot_multiple_bars(optimal_visited_states, visited_states_heuristic, "Number of visited states", "Visited States")
    plot_multiple_bars(optimal_queued_states, queued_states_heuristic, "Number of queued states", "Queued States")

