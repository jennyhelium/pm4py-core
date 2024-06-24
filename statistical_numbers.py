import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import math

import features


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
        curr = 0

        if i == "State Eq. LP Time":
            curr = row["State Eq. LP Solved LP"]
        elif i == "State Eq. ILP Time":
            curr = row["State Eq. ILP Solved LP"]
        elif i == "Ext. Eq. LP Time":
            curr = row["Ext. Eq. LP Solved LP"]
        elif i == "Ext. Eq. ILP Time":
            curr = row["Ext. Eq. ILP Solved LP"]

        if not isinstance(curr, str):  # curr is not Timeout
            num_lps += curr

        row_num += 1

    return num_lps


def replace_label_by_time(ls):
    for i in range(len(ls)):
        curr = ls[i].strip()
        if curr == "No Heuristic":
            ls[i] = "No Heuristic Time"
        elif curr == "Naive":
            ls[i] = "Naive Time"
        elif curr == "State Eq.":
            ls[i] = "State Eq. LP Time"
        elif curr == "State Eq. LP":
            ls[i] = "State Eq. LP Time"
        elif curr == "State Eq. ILP":
            ls[i] = "State Eq. ILP Time"
        elif curr == "Ext. Eq.":
            ls[i] = "Ext. Eq. LP Time"
        elif curr == "Ext. Eq. LP":
            ls[i] = "Ext. Eq. LP Time"
        elif curr == "Ext. Eq. ILP":
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

    for ind in range(len(df)):
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

        # if i == "No Heuristic":
        #   time += row["No Heuristic Time"]
        # elif i == "Naive":
        #   time += row["Naive Time"]
        # elif i == "State Eq. LP":
        #   time += row["State Eq. LP Time"]
        # elif i == "State Eq. ILP":
        #   time += row["State Eq. ILP Time"]
        # elif i == "Ext. Eq. LP":
        #   time += row["Ext. Eq. LP Time"]
        # elif i == "Ext. Eq. ILP":
        #   time += row["Ext. Eq. ILP Time"]
        if "No Heuristic" in i:
            time += row["No Heuristic Time"]
        elif "Naive" in i:
            time += row["Naive Time"]
        elif "State Eq. LP" in i:
            time += row["State Eq. LP Time"]
        elif "State Eq. ILP" in i:
            time += row["State Eq. ILP Time"]
        elif "Ext. Eq. LP" in i:
            time += row["Ext. Eq. LP Time"]
        elif "Ext. Eq. ILP" in i:
            time += row["Ext. Eq. ILP Time"]

        row_num += 1

    return round(time, 2)


def plot_multiple_bars(optimal, model, ls_heuristics, y_label, plot_title):
    x = ["Optimal", "Random", "Dijkstra", "Naive", "State LP", "State ILP", "Ext. LP", "Ext. ILP"]

    # y_optimal = [optimal for i in range(len_x)]

    # y_model = [model for i in range(len_x)]
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

    # if optimal == 0:
    # absolute difference
    # differences_model = [z - optimal for z in y_model]
    # differences_heuristics = [z - optimal for z in z_others]
    # else:  # percentage difference
    # differences_model = [round(z / optimal, 2) for z in y_model]
    # differences_heuristics = [round(z / optimal, 2) for z in z_others]
    len_x = len(x)
    x_axis = np.arange(len_x)

    bar_width = 0.25

    br1 = x_axis
    # br2 = [x + bar_width for x in br1]
    # br3 = [x + bar_width for x in br2]

    plt.bar(br1, values, bar_width)
    # plt.bar(br2, y_model, bar_width, label="model")
    # plt.bar(br3, z_others, bar_width, label="other heuristics")

    plt.axhline(optimal, color="green", linestyle="dashed", label="Optimal")
    if isinstance(model, list):
        plt.axhline(model[0], color="r", linestyle="dashed", label="Random")
        plt.axhline(model[1], color="b", linestyle="dashed", label="ML")
    else:
        plt.axhline(model, color="r", linestyle="dashed", label="Random")

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


def plot_multiple_bars_h(optimal, model, ls_heuristics, y_label, plot_title, model_name_ls=["Model"]):
    x = ["Optimal", "Model", "Dijkstra", "Naive", "State LP", "State ILP", "Ext. LP", "Ext. ILP"]
    y = ["Ext. ILP", "Ext. LP", "State ILP", "State LP", "Naive", "Dijkstra", "Random", "Optimal"]

    model_name_ls_reversed = model_name_ls
    model_name_ls_reversed.reverse()

    # y_optimal = [optimal for i in range(len_x)]

    # y_model = [model for i in range(len_x)]
    z_others = ls_heuristics

    values = []
    values.append(optimal)

    # values_y = []
    # values_y = values_y + ls_heuristics.reverse()

    if isinstance(model, list):
        for i in model:
            values.append(i)
            # values_y.append(len(model)-i)
        x = ["Optimal"] + model_name_ls + ["Dijkstra", "Naive", "State LP", "State ILP", "Ext. LP", "Ext. ILP"]
        y = ["Ext. ILP", "Ext. LP", "State ILP", "State LP", "Naive", "Dijkstra"] + model_name_ls_reversed + ["Optimal"]

    else:
        values.append(model)

    values = values + ls_heuristics
    values.reverse()
    # if optimal == 0:
    # absolute difference
    # differences_model = [z - optimal for z in y_model]
    # differences_heuristics = [z - optimal for z in z_others]
    # else:  # percentage difference
    # differences_model = [round(z / optimal, 2) for z in y_model]
    # differences_heuristics = [round(z / optimal, 2) for z in z_others]
    len_x = len(x)
    x_axis = np.arange(len_x)

    bar_width = 0.25

    br1 = x_axis

    plt.barh(br1, values, bar_width)

    plt.axvline(optimal, color="green", linestyle="dashed", label="Optimal")
    if isinstance(model, list):
        plt.axvline(model[0], color="r", linestyle="dashed", label=model_name_ls[1])
        plt.axvline(model[1], color="b", linestyle="dashed", label=model_name_ls[0])
    else:
        plt.axvline(model, color="r", linestyle="dashed", label="Random")

    # for i in range(len_x):
    #   plt.text(values[i] + 0.25, i, round(values[i], 3), ha="center")

    # plt.style.use('fivethirtyeight')
    plt.rcParams.update(plt.rcParamsDefault)

    plt.xscale("log")
    plt.yticks([r for r in range(len_x)], y)
    plt.xlabel(y_label)
    plt.ylabel("Comparison of optimal heuristics and only using one heuristic")
    plt.title(plot_title)
    plt.legend()
    plt.savefig("visualization/" + plot_title + ".png")
    plt.show()


def plot_multiple_bars_h_annot(optimal, model, ls_heuristics, y_label, plot_title, x_heuristics=None,
                               y_heuristics=None):
    if y_heuristics is None:
        y_heuristics = ["Ext. ILP", "Ext. LP", "State ILP", "State LP", "Naive", "Dijkstra",
                        "Model with recommendation function", "Model naive", "Optimal"]
    if x_heuristics is None:
        x_heuristics = ["Optimal", "Model naive", "Model with recommendation function", "Dijkstra",
                        "Naive", "State LP", "State ILP", "Ext. LP", "Ext. ILP"]
    x = x_heuristics
    y = y_heuristics

    # y_optimal = [optimal for i in range(len_x)]

    # y_model = [model for i in range(len_x)]
    z_others = ls_heuristics

    values = []
    values.append(optimal)

    # values_y = []
    # values_y = values_y + ls_heuristics.reverse()

    if isinstance(model, list):
        for i in model:
            values.append(i)
            # values_y.append(len(model)-i)
        # x = ["Optimal", "Random", "Model", "Dijkstra", "Naive", "State LP", "State ILP", "Ext. LP", "Ext. ILP"]
        # y = ["Ext. ILP", "Ext. LP", "State ILP", "State LP", "Naive", "Dijkstra", "Model", "Random", "Optimal"]

    else:
        values.append(model)

    values = values + ls_heuristics
    values.reverse()

    len_x = len(x)
    x_axis = np.arange(len_x)

    bar_width = 0.25

    br1 = x_axis

    plt.barh(br1, values, bar_width)

    plt.axvline(optimal, color="green", linestyle="dashed", label="Optimal")
    if isinstance(model, list):
        plt.axvline(model[0], color="r", linestyle="dashed", label="Random")
        plt.axvline(model[1], color="b", linestyle="dashed", label="ML")
    else:
        plt.axvline(model, color="r", linestyle="dashed", label="Random")

    # for i in range(len_x):
    #   plt.text(values[i] + 0.25, i, round(values[i], 3), ha="center")

    # plt.style.use('fivethirtyeight')
    plt.rcParams.update(plt.rcParamsDefault)

    plt.xscale("log")
    plt.yticks([r for r in range(len_x)], y)
    plt.xlabel(y_label)
    plt.ylabel("Comparison of optimal heuristics and only using one heuristic")
    plt.title(plot_title)
    plt.legend()
    plt.savefig("visualization/" + plot_title + ".png")
    plt.show()


def return_model_metrics(x_test, predictions, timeout):
    """
    return evaluation values
    Parameters
    ----------
    x_test
    predictions

    Returns
    -------

    """
    model_idx = replace_label_by_time(predictions)
    model_time = time_using_model(x_test, model_idx)
    model_timeouts, _ = timeouts_optimal_heuristics(x_test, timeout, model_idx)
    model_lps = lps_optimal_heuristics(x_test, model_idx)
    model_queued_states = states_optimal_heuristics(x_test, False, model_idx)

    return model_idx, model_time, model_timeouts, model_lps, model_queued_states


def evaluate_time(df, time_ls, model_name_ls):
    optimal_time = time_with_optimal_heuristics(df)
    time_heuristics = time_using_one_heuristic(df)

    plot_multiple_bars_h(optimal_time, time_ls, time_heuristics, "Time in seconds",
                         "Computation Time", model_name_ls)


def evaluate_timeouts(df, time_ls, model_name_ls, timeout):
    optimal_timeouts, _ = timeouts_optimal_heuristics(df, timeout)

    timeouts_heuristics = []

    heuristics = ["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time", "Ext. Eq. LP Time",
                  "Ext. Eq. ILP Time"]

    for h in heuristics:
        num_timeouts = len(df[df[h].ge(timeout)])
        timeouts_heuristics.append(num_timeouts)

    plot_multiple_bars_h(optimal_timeouts, time_ls, timeouts_heuristics, "Number of timeouts",
                         "Timeouts", model_name_ls)


def evaluate_lps(df, time_ls, model_name_ls):
    optimal_lps = lps_optimal_heuristics(df)
    lps_one_heuristic = lps_using_one_heuristic(df)

    plot_multiple_bars_h(optimal_lps, time_ls, lps_one_heuristic, "Number of solved lps",
                         "Solved LPs", model_name_ls)


def evaluate_queued(df, time_ls, model_name_ls):
    optimal_states = states_optimal_heuristics(df, False)
    queued_states_heuristic = states_one_heuristic(df, False)

    plot_multiple_bars_h(optimal_states, time_ls, queued_states_heuristic, "Number of queued states",
                         "Queued States", model_name_ls)


def get_winners_grouped_by_heuristic(df, timeout):
    min_index = df[["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time",
                    "Ext. Eq. LP Time", "Ext. Eq. ILP Time"]].idxmin(axis="columns")

    row_num = 0

    no_heuristic = []
    naive = []
    state_lp = []
    state_ilp = []
    ext_lp = []
    ext_ilp = []

    for i in min_index:
        row = df.iloc[row_num, :]
        curr = 0

        if i == "No Heuristic Time":
            no_heuristic.append(row)
        elif i == "Naive Time":
            naive.append(row)
        elif i == "State Eq. LP Time":
            state_lp.append(row)
        elif i == "State Eq. ILP Time":
            state_ilp.append(row)
        elif i == "Ext. Eq. LP Time":
            ext_lp.append(row)
        elif i == "Ext. Eq. ILP Time":
            ext_ilp.append(row)

        row_num += 1

    return no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp


def get_avg_trace_len_winners(no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp):
    groups = [no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp]

    val_ls = []

    for g in groups:
        val = 0
        if g:
            for i in g:
                val += len(i["Trace"])
            val = val / len(g)

        val_ls.append(val)

    return val_ls


def get_num_transitions_in_model(pn):
    return len(pn.transitions)


# alignment costs?
# fitness
# statistics
# len trace, model
def get_avg_value_winners(no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp, value="fitness"):
    groups = [no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp]

    val_ls = []

    for g in groups:
        val = 0
        if g:
            heuristic = ""
            if g == no_heuristic:
                heuristic = "No Heuristic"
            elif g == naive:
                heuristic = "Naive"
            elif g == state_lp:
                heuristic = "State Eq. LP"
            elif g == state_ilp:
                heuristic = "State Eq. ILP"
            elif g == ext_lp:
                heuristic = "Ext Eq. LP"
            elif g == ext_ilp:
                heuristic = "Ext Eq. ILP"

            for i in g:
                val += i[heuristic][value]
            val = val / len(g)
        # val = val

        val_ls.append(val)

    return val_ls


def heuristics_values(df: pd.DataFrame, v="fitness"):
    heuristics = ["No Heuristic", "Naive", "State Eq. LP", "State Eq. ILP", "Ext. Eq. LP", "Ext. Eq. ILP"]
    values = []
    for h in heuristics:
        value = 0

        for i in range(len(df)):
            curr = df[h][i]
            if curr is not None:
                value += curr[v]

        value /= len(df)
        values.append(value)

    return values


def draw_box_plot_winners_apply(no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp, apply, use_pn, use_trace):
    groups = [no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp]

    val_ls = []

    for g in groups:
        val = []
        if g:
            heuristic = ""
            if g == no_heuristic:
                heuristic = "No Heuristic"
            elif g == naive:
                heuristic = "Naive"
            elif g == state_lp:
                heuristic = "State Eq. LP"
            elif g == state_ilp:
                heuristic = "State Eq. ILP"
            elif g == ext_lp:
                heuristic = "Ext Eq. LP"
            elif g == ext_ilp:
                heuristic = "Ext Eq. ILP"

            for i in g:
                trace = i["Trace"]
                pn = i["Petri Net"]

                if use_pn and use_trace:
                    curr = apply(pn, trace)
                elif use_trace:
                    curr = apply(trace)
                elif use_pn:
                    curr = apply(pn)
                if isinstance(curr, (list, tuple)):
                    val.append(curr[0])
                else:
                    val.append(curr)

        val_ls.append(np.asarray(val))

    fig = plt.figure(figsize=(10, 7))

    # Creating axes instance
    # ax = fig.add_axes([0, 0, 1, 1])

    # Creating plot
    # bp = ax.boxplot(val_ls)
    plt.boxplot(val_ls)
    plt.xticks([1, 2, 3, 4, 5, 6], ["No", "Naive", "State LP", "State ILP", "Ext LP", "Ext ILP"])

    # show plot
    plt.show()


def draw_box_plot_winners(no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp, value="fitness"):
    groups = [no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp]

    val_ls = []

    for g in groups:
        val = []
        if g:
            heuristic = ""
            if g == no_heuristic:
                heuristic = "No Heuristic"
            elif g == naive:
                heuristic = "Naive"
            elif g == state_lp:
                heuristic = "State Eq. LP"
            elif g == state_ilp:
                heuristic = "State Eq. ILP"
            elif g == ext_lp:
                heuristic = "Ext Eq. LP"
            elif g == ext_ilp:
                heuristic = "Ext Eq. ILP"

            for i in g:
                val.append(i[heuristic][value])

        val_ls.append(np.asarray(val))

    fig = plt.figure(figsize=(10, 7))

    # Creating axes instance
    # ax = fig.add_axes([0, 0, 1, 1])

    # Creating plot
    # bp = ax.boxplot(val_ls)
    plt.boxplot(val_ls)
    plt.xticks([1, 2, 3, 4, 5, 6], ["No", "Naive", "State LP", "State ILP", "Ext LP", "Ext ILP"])

    # show plot
    plt.show()


def draw_box_plot_all(df, v="fitness"):
    heuristics = ["No Heuristic", "Naive", "State Eq. LP", "State Eq. ILP", "Ext. Eq. LP", "Ext. Eq. ILP"]
    values = []
    for h in heuristics:
        value = []

        for i in range(len(df)):
            curr = df[h][i]
            if curr is not None:
                value.append(curr[v])

        values.append(np.asarray(value))

    fig = plt.figure(figsize=(10, 7))

    # Creating axes instance
    # ax = fig.add_axes([0, 0, 1, 1])

    # Creating plot
    # bp = ax.boxplot(values)
    plt.boxplot(values)
    plt.xticks([1, 2, 3, 4, 5, 6], ["No", "Naive", "State LP", "State ILP", "Ext LP", "Ext ILP"])
    # show plot
    plt.show()


if __name__ == "__main__":
    # df = pd.read_pickle("results/sepsis_filtered_inductive_0_3.pkl")
    # df = pd.read_pickle("results/road_inductive_02_updated.pkl")
    # df = pd.read_pickle("results/road_filtered_inductive_0_3.pkl")
    # df = pd.read_pickle("results/road_heuristic_3.pkl")

    df = pd.read_pickle("results/sepsis_alpha_0.2_3.pkl")
    # df = pd.read_pickle("results/sepsis_inductive_0_3.pkl")
    # df = pd.read_pickle("results/sepsis_filtered_inductive_0_3.pkl") Fehler?

    # df = pd.read_pickle("results/prepaid_inductive_0_3.pkl")
    # df = pd.read_pickle("results/prepaid2024-03-08 00:24:01.pkl")
    # df = pd.read_pickle("results/prepaid_filtered_inductive_0_3.pkl")

    len_df = len(df)
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
        num_timeouts = len(df[df[h].ge(timeout)])
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

    plot_multiple_bars_h(optimal_time, random_time, times_one_heuristic, "Time in seconds", "Computation times")
    plot_multiple_bars_h(optimal_queued_states, random_queued_states, queued_states_heuristic,
                         "Number of queued states",
                         "Queued States")
    plot_multiple_bars(optimal_time, random_time, times_one_heuristic, "Time in seconds", "Computation times")
    plot_multiple_bars(optimal_timeouts, random_timeouts, timeouts_heuristics, "Number of timeouts",
                       plot_title="Timeouts")
    plot_multiple_bars(optimal_lps, random_lps, lps_one_heuristic, "Number of solved lps", "Solved LPs")
    plot_multiple_bars(optimal_visited_states, random_visited_states, visited_states_heuristic,
                       "Number of visited states", "Visited States")

    no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp = get_winners_grouped_by_heuristic(df, 180)
    print("Avg fitness winners ", get_avg_value_winners(no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp,
                                                        value="fitness"))
    print("Avg costs winners ",
          get_avg_value_winners(no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp, value="cost"))
    print(heuristics_values(df))
    print(heuristics_values(df, "cost"))
    print("Avg trace length winners ",
          get_avg_trace_len_winners(no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp))
    draw_box_plot_winners_apply(no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp,
                                features.parallelism_model_multiplied, use_pn=True, use_trace=False)
    draw_box_plot_winners_apply(no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp,
                                features.choice, use_pn=True, use_trace=False)
    draw_box_plot_winners_apply(no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp,
                                features.len_trace, use_pn=False, use_trace=True)
    # draw_box_plot_winners_apply(no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp,
    #                           features.parallelism_model_multiplied, use_pn=True, use_trace=False)
    # boxplots fitness
    draw_box_plot_winners(no_heuristic, naive, state_lp, state_ilp, ext_lp, ext_ilp)
    draw_box_plot_all(df)
