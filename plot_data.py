import sys
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px


def addlabels(x, y, z):
    for i in range(len(x)):
        plt.text(i, y[i], z[i], ha='center')


def distribution_winners(df, timeout):
    min_val_idx = df[["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time",
                      "Ext. Eq. LP Time", "Ext. Eq. ILP Time"]].idxmin(axis="columns")

    count_no = 0
    count_naive = 0
    count_state_lp = 0
    count_state_ilp = 0
    count_ext_lp = 0
    count_ext_ilp = 0

    row_num = 0

    for i in min_val_idx:
        row = df.iloc[row_num, :]
        min_time = row[i]

        if min_time < timeout:
            if i == "No Heuristic Time":
                count_no = count_no + 1
            elif i == "Naive Time":
                count_naive = count_naive + 1
            elif i == "State Eq. LP Time":
                count_state_lp = count_state_lp + 1
            elif i == "State Eq. ILP Time":
                count_state_ilp = count_state_ilp + 1
            elif i == "Ext. Eq. LP Time" or i == "Ext. State Eq. LP Time":
                count_ext_lp = count_ext_lp + 1
            elif i == "Ext. Eq. ILP Time" or i == "Ext. State Eq. ILP Time":
                count_ext_ilp = count_ext_ilp + 1

        row_num = row_num + 1

    return count_no, count_naive, count_state_lp, count_state_ilp, count_ext_lp, count_ext_ilp


def create_sunburst_plot(df, timeout):
    count_no, count_naive, count_state_lp, count_state_ilp, count_ext_lp, count_ext_ilp = distribution_winners(df,
                                                                                                               timeout)
    data_curr = dict(value=[count_no, count_naive, count_state_lp, count_state_ilp, count_ext_lp, count_ext_ilp],
                     heuristic=["No Heuristic", "Naive", "State Equation", "State Equation", "Extended State Eq.",
                                "Extended State Eq."],
                     parent=[None, None, "State Eq. LP", "State Eq. ILP", "Ext. LP", "Ext. ILP"]
                     )

    df = pd.DataFrame(data_curr)

    print(df)

    fig = px.sunburst(df, path=["heuristic", "parent"], values="value",color="heuristic",
                      color_discrete_map={"State Equation": "#636EFA", "No Heuristic": "red", "Naive": "#00CC96",
                                          "Extended State Eq.": "#FFA15A"})
    fig.show()


def create_bar_plot(df, timeout):
    count_no, count_naive, count_state_lp, count_state_ilp, count_ext_lp, count_ext_ilp = distribution_winners(df,
                                                                                                               timeout)

    data_plot = {"No Heuristic": count_no, "Naive": count_naive, "State LP": count_state_lp,
                 "State ILP": count_state_ilp, "Ext. LP": count_ext_lp, "Ext. ILP": count_ext_ilp}

    heuristics_plot = list(data_plot.keys())
    values = list(data_plot.values())
    sum_values = sum(values)
    percentages = [str(round(v / sum_values * 100, 2)) + "%" for v in values]

    # create plot
    plt.bar(heuristics_plot, values, color='maroon', width=0.4)

    addlabels(heuristics_plot, values, percentages)

    plt.xlabel("Heuristics")
    plt.ylabel("No. of traces")
    plt.title("No. of each heuristic with minimal computation time")
    plt.show()


def create_box_plot(df):
    df_times = df[["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time",
                   "Ext. Eq. LP Time", "Ext. Eq. ILP Time"]]

    df_times.plot(
        kind='box',
        subplots=True,
        sharey=False,
        figsize=(10, 6)
    )

    # increase spacing between subplots
    plt.subplots_adjust(wspace=1.5)
    plt.show()


def create_line_plot(df):
    df_times = df[["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time",
                   "Ext. Eq. LP Time", "Ext. Eq. ILP Time"]]

    df_times.plot(
        kind='line'
    )

    plt.show()


# data = pd.read_pickle("results/domestic_filtered_inductive_0_3.pkl")

data = pd.read_pickle("results/road_heuristic_3.pkl")
# data = pd.read_pickle("permit_inductive_0,2_curr.pkl")

# data = pd.read_pickle("results/prepaid2024-03-08 00:24:01.pkl")
#
# data = pd.read_pickle("results/request2024-03-06 02:17:46.pkl")
# data = pd.read_pickle("results/domestic_inductive_3.pkl")
# data = pd.read_pickle("permit_curr.pkl")
# data = pd.read_pickle("results/road_im_2_noise(0.2)_2024-02-24 08:59:45.pkl")

create_sunburst_plot(data, 180)
create_bar_plot(data, 180)
create_line_plot(data)
create_box_plot(data)
