import pandas as pd


def get_var_name(var):
    for name, value in globals().items():
        if value is var:
            return name


def update_ext_columns(df1: pd.DataFrame, df2: pd.DataFrame):
    name_df = get_var_name(df1)

    updated_df = df1.copy()
    updated_df['Ext. Eq. LP'] = df2['Ext. Eq. LP']
    updated_df['Ext. Eq. ILP'] = df2['Ext. Eq. ILP']
    updated_df['Ext. Eq. LP Time'] = df2['Ext. Eq. LP Time']
    updated_df['Ext. Eq. ILP Time'] = df2['Ext. Eq. ILP Time']
    updated_df['Ext. Eq. LP Solved LP'] = df2['Ext. Eq. LP Solved LP']
    updated_df['Ext. Eq. ILP Solved LP'] = df2['Ext. Eq. ILP Solved LP']

    updated_df.to_pickle("results/" + name_df + "_updated_ext.pkl")


if __name__ == "__main__":
    """
    df = pd.read_pickle("results/road_im_2_noise(0.2)_2024-02-24 08:59:45.pkl")

    road_top_10 = pd.read_pickle("results/road_top_10_inductive_0_3.pkl")
    road_top_5 = pd.read_pickle("results/road_top_5_inductive_0_3.pkl")
    road_top_1 = pd.read_pickle("results/road_top_1_inductive_0_3.pkl")
    road_top_1_ext = pd.read_pickle("results/road_top_1_ext_inductive_0_3.pkl")
    road_top_5_ext = pd.read_pickle("results/road_top_5_ext_inductive_0_3.pkl")
    road_top_10_ext = pd.read_pickle("results/road_top_10_ext_inductive_0_3.pkl")

    update_ext_columns(road_top_1, road_top_1_ext)
    update_ext_columns(road_top_10, road_top_10_ext)
    update_ext_columns(road_top_5, road_top_5_ext)

    sepsis_top_1 = pd.read_pickle("results/sepsis_top_1_inductive_0_3.pkl")
    sepsis_top_1_ext = pd.read_pickle("results/sepsis_top_1_ext_inductive_0_3.pkl")

    update_ext_columns(sepsis_top_1, sepsis_top_1_ext)

    prepaid_top_10 = pd.read_pickle("results/prepaid_top_10_inductive_0_3.pkl")
    prepaid_top_5 = pd.read_pickle("results/prepaid_top_5_inductive_0_3.pkl")
    prepaid_top_1 = pd.read_pickle("results/prepaid_top_1_inductive_0_3.pkl")
    prepaid_top_1_ext = pd.read_pickle("results/prepaid_top_1_ext_inductive_0_3.pkl")
    prepaid_top_5_ext = pd.read_pickle("results/prepaid_top_5_ext_inductive_0_3.pkl")
    prepaid_top_10_ext = pd.read_pickle("results/prepaid_top_10_ext_inductive_0_3.pkl")

    update_ext_columns(prepaid_top_1, prepaid_top_1_ext)
    update_ext_columns(prepaid_top_5, prepaid_top_5_ext)
    update_ext_columns(prepaid_top_10, prepaid_top_10_ext)

    request_top_10 = pd.read_pickle("results/request_top_10_inductive_0_3.pkl")
    request_top_5 = pd.read_pickle("results/request_top_5_inductive_0_3.pkl")
    request_top_1 = pd.read_pickle("results/request_top_1_inductive_0_3.pkl")
    request_top_10_ext = pd.read_pickle("results/request_top_10_ext_inductive_0_3.pkl")
    request_top_5_ext = pd.read_pickle("results/request_top_5_ext_inductive_0_3.pkl")
    request_top_1_ext = pd.read_pickle("results/request_top_1_ext_inductive_0_3.pkl")

    update_ext_columns(request_top_1, request_top_1_ext)
    update_ext_columns(request_top_5, request_top_5_ext)
    update_ext_columns(request_top_10, request_top_10_ext)

    domestic_top_10 = pd.read_pickle("results/domestic_top_10_inductive_0_3.pkl")
    domestic_top_5 = pd.read_pickle("results/domestic_top_5_inductive_0_3.pkl")
    domestic_top_1 = pd.read_pickle("results/domestic_top_1_inductive_0_3.pkl")
    domestic_top_10_ext = pd.read_pickle("results/domestic_top_10_ext_inductive_0_3.pkl")
    domestic_top_5_ext = pd.read_pickle("results/domestic_top_5_ext_inductive_0_3.pkl")
    domestic_top_1_ext = pd.read_pickle("results/domestic_top_1_ext_inductive_0_3.pkl")

    update_ext_columns(domestic_top_1, domestic_top_1_ext)
    update_ext_columns(domestic_top_5, domestic_top_5_ext)
    update_ext_columns(domestic_top_10, domestic_top_10_ext)

    international_top_10 = pd.read_pickle("results/international_declaration_top_10_inductive_0_3.pkl")
    international_top_5 = pd.read_pickle("results/international_declaration_top_5_inductive_0_3.pkl")
    international_top_1 = pd.read_pickle("results/international_declaration_top_1_inductive_0_3.pkl")
    international_top_10_ext = pd.read_pickle("results/international_declaration_top_10_ext_inductive_0_3.pkl")
    international_top_5_ext = pd.read_pickle("results/international_declaration_top_5_ext_inductive_0_3.pkl")
    international_top_1_ext = pd.read_pickle("results/international_declaration_top_1_ext_inductive_0_3.pkl")

    update_ext_columns(international_top_1, international_top_1_ext)
    update_ext_columns(international_top_5, international_top_5_ext)
    update_ext_columns(international_top_10, international_top_10_ext)
    
    sepsis_top_5 = pd.read_pickle("results/sepsis_top_5_inductive_0_3.pkl")
    sepsis_top_5_ext = pd.read_pickle("results/sepsis_top_5_ext_inductive_0_3.pkl")
    update_ext_columns(sepsis_top_5, sepsis_top_5_ext)
    
    bpi12_top_1 = pd.read_pickle("results/bpi12_top_1_inductive_0_3.pkl")
    bpi_top_1_ext = pd.read_pickle("results/bpi12_top_1_ext_inductive_0_3.pkl")
    update_ext_columns(bpi12_top_1, bpi_top_1_ext)
    
    road_inductive_02 = pd.read_pickle("results/road_inductive_3.pkl")
    road_inductivev_02_ext = pd.read_pickle("results/road_ext_inductive_0.2_3.pkl")
    update_ext_columns(road_inductive_02, road_inductivev_02_ext)
    
    sepsis_inductive_02 = pd.read_pickle("results/sepsis_inductive_02_updated.pkl")
    sepsis_inductive_02_ext = pd.read_pickle("results/sepsis_ext_inductive_0.2_3.pkl")
    update_ext_columns(sepsis_inductive_02, sepsis_inductive_02_ext)
      """
    prGm6 = pd.read_pickle("results/prGm6_no_3.pkl")
    prGm6_ext = pd.read_pickle("results/prGm6_ext_no_0_3.pkl")
    update_ext_columns(prGm6, prGm6_ext)

    #bpi12_top_5 = pd.read_pickle("results")
