import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator

import features
import plot_data



def get_merged_data():
    road_inductive_0 = pd.read_pickle("results/road_inductive_0_3.pkl")
    road_inductive_02 = pd.read_pickle("results/road_inductive_3.pkl")
    road_heuristic = pd.read_pickle("results/road_heuristic_3.pkl")
    road_filtered = pd.read_pickle("results/road_filtered_inductive_0_3.pkl")

    sepsis_alpha = pd.read_pickle("results/sepsis_alpha_0.2_3.pkl")
    sepsis_inductive_02 = pd.read_pickle("results/sepsis_inductive_02_updated.pkl")
    sepsis_filtered = pd.read_pickle("results/sepsis_filtered_inductive_0_3.pkl")

    italian_alpha = pd.read_pickle("results/italian_alpha_0.2_3.pkl")

    # bpi12
    bpi12_inductive_02_curr = pd.read_pickle("results/bpi12_inductive_0.2_curr.pkl")

    # bpi13
    prGm6 = pd.read_pickle("results/prGm6_no_3.pkl")
    prFm6 = pd.read_pickle("prFm6no_curr.pkl")

    # BPI20
    prepaid_inductive_0 = pd.read_pickle("results/prepaid_inductive_0_3.pkl")
    prepaid_inductive_02 = pd.read_pickle("results/prepaid2024-03-08 00:24:01.pkl")
    prepaid_filtered = pd.read_pickle("results/prepaid_filtered_inductive_0_3.pkl")

    request_inductive_0 = pd.read_pickle("results/request_inductive_0_3.pkl")
    request_inductive_02 = pd.read_pickle("results/request2024-03-06 02:17:46.pkl")
    request_filtered = pd.read_pickle("results/request_filtered_inductive_0_3.pkl")

    domestic_inductive_0 = pd.read_pickle("results/domestic_inductive_0_3.pkl")
    domestic_inductive_02 = pd.read_pickle("results/domestic_inductive_3.pkl")
    domestic_filtered = pd.read_pickle("results/domestic_filtered_inductive_0_3.pkl")

    international_inductive_0 = pd.read_pickle("results/international_declaration_inductive_0_3.pkl")
    international_inductive_02 = pd.read_pickle("results/international_declaration_inductive_curr.pkl")
    international_filtered = pd.read_pickle("results/international_declaration_filtered_inductive_0_3.pkl")

    permit_inductive_0 = pd.read_pickle("results/permit_inductive_0_curr.pkl")
    permit_inductive_02 = pd.read_pickle("permit_inductive_0.2_curr.pkl")

    data = road_inductive_02
    data = data._append(road_inductive_0)
    data = data._append(road_heuristic)
    data = data._append(road_filtered)
    data = data._append(sepsis_alpha)
    data = data._append(sepsis_inductive_02)
    # data = data._append(sepsis_filtered)
    data = data._append(italian_alpha)
    data = data._append(bpi12_inductive_02_curr)
    data = data._append(prGm6)
    data = data._append(prFm6)
    data = data._append(prepaid_inductive_0)
    data = data._append(prepaid_inductive_02)
    data = data._append(prepaid_filtered)
    data = data._append(request_inductive_0)
    data = data._append(request_inductive_02)
    data = data._append(request_filtered)
    data = data._append(domestic_inductive_0)
    data = data._append(domestic_inductive_02)
    data = data._append(domestic_filtered)
    data = data._append(international_inductive_0)
    data = data._append(international_inductive_02)
    data = data._append(international_filtered)
    data = data._append(permit_inductive_0)
    data = data._append(permit_inductive_02)

    return data


def create_features(trace, pn):
    curr_features = []

    matches_traces, matches_traces_ratio, matches_models, matches_models_ratio = features.matching_labels(pn, trace)
    curr_features.append(matches_traces)
    curr_features.append(matches_traces_ratio)
    curr_features.append(matches_models)
    curr_features.append(matches_models_ratio)

    len_trace, trace_transitions_ratio, trace_places_ratio, transitions_trace_ratio, places_trace_ratio = features.trace_ratio(pn, trace)
    curr_features.append(len_trace)
    curr_features.append(trace_transitions_ratio)
    curr_features.append(trace_places_ratio)
    curr_features.append(transitions_trace_ratio)
    curr_features.append(places_trace_ratio)

    trace_loop, trace_loop_ratio, max_reps, max_reps_ratio, mean_reps, mean_reps_ratio, sum_reps, sum_reps_ratio = features.trace_loop(
        trace)
    curr_features.append(trace_loop)
    curr_features.append(trace_loop_ratio)
    curr_features.append(max_reps)
    curr_features.append(max_reps_ratio)
    curr_features.append(mean_reps)
    curr_features.append(mean_reps_ratio)
    curr_features.append(sum_reps)
    curr_features.append(sum_reps_ratio)

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

    return curr_features


# data = merge_data.get_merged_data()


def get_features_data(data):
    # Encode data
    # Schnellste Heuristic Time = 1, else 0

    # multi class classification
    # label column
    plot_data.create_sunburst_plot(data, 180)

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
    encode_times = encode_times[encode_times.Label != "Timeout"]

    # one_hot_encoded = pd.get_dummies(encode_times, columns=["Label"])

    features_data = []
    feature = []

    start_features = time.time()

    for i in range(len(encode_times)):
        row = encode_times.iloc[i, :]
        trace = row["Trace"]
        pn = row["Petri Net"]
        im = row["Initial Marking"]
        fm = row["Final Marking"]

        row_features = create_features(trace, pn)

        len_features = len(row_features)

        feature.append(row_features)

    end_features = time.time()
    time_features = end_features - start_features

    print("Time to compute features: ", time_features)

    features_data.append(feature)

    columns = ["Matching Labels Trace", "Matching Trace Ratio", "Matching Labels Models",
               "Matching Model Ratio", "Trace Length", "Trace Trans Ratio",
               "Trace Place Ratio", "Trans Trace Ratio", "Place Trace Ratio",
               "Trace Loop", "Trace Loop Ratio", "Trace Loop Max",
               "Trace Loop Max Ratio", "Trace Loop Mean", "Trace Loop Mean Ratio",
               "Trace Loop Sum", "Trace Loop Sum Ratio", "Model Duplicates",
               "Model Duplicates Ratio", "Trans No In-arc", "Trans No In-arc ratio",
               "Silent Transitions", "Silent Transitions ratio", "Parallelism Sum",
               "Parallelism Sum Ratio", "Parallelism Mult", "Parallelism Mult Ratio",
               "Choice Sum", "Choice Sum Ratio", "Choice Mult", "Choice Mult Ratio",
               "Simplicity"]

    features_df = pd.DataFrame(feature, columns=columns, dtype=float)

    # scale features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features_df)
    scaled_df = pd.DataFrame(scaled_features, columns=features_df.columns)

    result = encode_times.join(scaled_df)

    print(result)
    return result, len_features
def get_train_test_set(data, ratio=0.8):

    # create train test split
    data = data.sample(frac=1, random_state=1)  # shuffle data

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

if __name__ == "__main__":
    data = get_merged_data()
    data_features, len_features = get_features_data(data)
