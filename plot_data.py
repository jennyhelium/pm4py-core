import sys
import pandas as pd
import matplotlib as plt


def create_bar_plot(df):
    maxValIndex = df[["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time",
                      "Ext. State Eq. LP Time", "Ext. State Eq. ILP Time"]].idxmin(axis="columns")
    count_no = 0
    count_naive = 0
    count_state_lp = 0
    count_state_ilp = 0
    count_ext_lp = 0
    count_ext_ilp = 0

    for i in maxValIndex:
        if i == "No Heuristic Time":
            count_no = count_no + 1
        elif i == "Naive Time":
            count_naive = count_naive + 1
        elif i == "State Eq. LP Time":
            count_state_lp = count_state_lp + 1
        elif i == "State Eq. ILP Time":
            count_state_ilp = count_state_ilp + 1
        elif i == "Ext. State Eq. LP Time":
            count_ext_lp = count_ext_lp + 1
        else:
            count_ext_ilp = count_ext_ilp + 1

    data_plot = {"No Heuristic": count_no, "Naive": count_naive, "State LP": count_state_lp,
                 "State ILP": count_state_ilp, "Ext. LP": count_ext_lp, "Ext. ILP": count_ext_ilp}

    heuristics_plot = list(data_plot.keys())
    values = list(data_plot.values())

    # create plot
    plt.bar(heuristics_plot, values, color='maroon', width=0.4)

    plt.xlabel("Heuristics")
    plt.ylabel("No. of traces")
    plt.title("No. of each heuristic with minimal computation time")
    plt.show()


def create_box_plot(df):
    df_times = df[["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time",
                   "Ext. State Eq. LP Time", "Ext. State Eq. ILP Time"]]

    df_times.plot(
        kind='box',
        subplots=True,
        sharey=False,
        figsize=(10, 6)
    )

    # increase spacing between subplots
    plt.subplots_adjust(wspace=1.5)
    plt.show()

data = pd.read_pickle("prGm6no_curr.pkl")
create_bar_plot(data)
create_box_plot(data)