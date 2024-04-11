import pandas as pd

if __name__ == "__main__":
    df = pd.read_pickle("results/sepsis_inductive_2_2024-02-24 08:55:06.pkl")

    # evt. change column names ['Ext. State Eq. LP Time', 'Ext. State Eq. ILP Time']
    # to ['Ext. Eq. LP Time', 'Ext. Eq. ILP Time']
    # and add missing columns for solved lps
    if 'Ext. State Eq. LP Time' in df.columns:
        df.rename(
            columns={'Ext. State Eq. LP': 'Ext. Eq. LP', 'Ext. State Eq. ILP': 'Ext. Eq. ILP',
                     'Ext. State Eq. LP Time': 'Ext. Eq. LP Time', 'Ext. State Eq. ILP Time': 'Ext. Eq. ILP Time'},
            inplace=True)

        column_lps = ["State Eq. LP Solved LP", "State Eq. ILP Solved LP", "Ext. Eq. LP Solved LP",
                      "Ext. Eq. ILP Solved LP"]
        h_lps = ["State Eq. LP", "State Eq. ILP", "Ext. Eq. LP", "Ext. Eq. ILP"]

        lps = [[] for i in range(len(h_lps))]

        for ind in df.index:
            row = df.iloc[ind, :]

            for i in range(len(h_lps)):
                alignment = row[h_lps[i]]

                if alignment is None:
                    lps[i].append("Timeout")
                else:
                    lps[i].append(alignment["lp_solved"])

        for i in range(len(column_lps)):
            df[column_lps[i]] = lps[i]

        print(df)

        df.to_pickle("results/" + "sepsis_inductive_02" + "_updated" + ".pkl")
