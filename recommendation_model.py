import pandas as pd
import numpy as np
import time
import random
import math
import matplotlib.pyplot as plt
from pm4py.algo.evaluation.simplicity import algorithm as simplicity_evaluator
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

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
    road_inductive_0 = pd.read_pickle("results/road_inductive_0_3.pkl")
    road_inductive_02 = pd.read_pickle("results/road_inductive_3.pkl")
    road_heuristic = pd.read_pickle("results/road_heuristic_3.pkl")
    road_filtered = pd.read_pickle("results/road_filtered_inductive_0_3.pkl")
    road_max_cov = pd.read_pickle("results/road_max_cov_filtered_inductive_0_3.pkl")

    sepsis_alpha = pd.read_pickle("results/sepsis_alpha_0.2_3.pkl")
    sepsis_inductive_02 = pd.read_pickle("results/sepsis_inductive_02_updated.pkl")
    sepsis_filtered = pd.read_pickle("results/sepsis_filtered_inductive_0_3.pkl")
    sepsis_max_cov = pd.read_pickle("results/sepsis_max_cov_filtered_inductive_0_3.pkl")


    italian_alpha = pd.read_pickle("results/italian_alpha_0.2_3.pkl")

    # bpi12
    bpi12_inductive_02_curr = pd.read_pickle("results/bpi12_inductive_0.2_curr.pkl")
    bpi12_filtered = pd.read_pickle("results/bpi12_filtered_inductive_0_3.pkl")

    # bpi13
    prGm6 = pd.read_pickle("results/prGm6_no_3.pkl")
    prFm6 = pd.read_pickle("prFm6no_curr.pkl")

    # BPI20
    prepaid_inductive_0 = pd.read_pickle("results/prepaid_inductive_0_3.pkl")
    prepaid_inductive_02 = pd.read_pickle("results/prepaid2024-03-08 00:24:01.pkl")
    prepaid_filtered = pd.read_pickle("results/prepaid_filtered_inductive_0_3.pkl")
    prepaid_max_cov = pd.read_pickle("results/prepaid_max_cov_filtered_inductive_0_3.pkl")

    request_inductive_0 = pd.read_pickle("results/request_inductive_0_3.pkl")
    request_inductive_02 = pd.read_pickle("results/request2024-03-06 02:17:46.pkl")
    request_filtered = pd.read_pickle("results/request_filtered_inductive_0_3.pkl")
    request_max_cov = pd.read_pickle("results/request_max_cov_filtered_inductive_0_3.pkl")

    domestic_inductive_0 = pd.read_pickle("results/domestic_inductive_0_3.pkl")
    domestic_inductive_02 = pd.read_pickle("results/domestic_inductive_3.pkl")
    domestic_filtered = pd.read_pickle("results/domestic_filtered_inductive_0_3.pkl")
    domestic_max_cov = pd.read_pickle("results/domestic_max_cov_filtered_inductive_0_3.pkl")

    international_inductive_0 = pd.read_pickle("results/international_declaration_inductive_0_3.pkl")
    international_inductive_02 = pd.read_pickle("results/international_declaration_inductive_curr.pkl")
    international_filtered = pd.read_pickle("results/international_declaration_filtered_inductive_0_3.pkl")
    international_max_cov = pd.read_pickle("results/international_declaration_max_cov_filtered_inductive_0_3.pkl")

    permit_inductive_0 = pd.read_pickle("results/permit_inductive_0_curr.pkl")
    permit_inductive_02 = pd.read_pickle("permit_inductive_0.2_curr.pkl")
    permit_filtered_curr = pd.read_pickle("results/permit_inductive_0_curr.pkl")

    data = road_inductive_02
    data = data._append(road_inductive_0)
    data = data._append(road_heuristic)
    data = data._append(road_filtered)
    data = data._append(road_max_cov)

    data = data._append(sepsis_alpha)
    data = data._append(sepsis_inductive_02)
    data = data._append(sepsis_filtered)
    data = data._append(sepsis_max_cov)

    data = data._append(italian_alpha)

    data = data._append(bpi12_inductive_02_curr)
    data = data._append(bpi12_filtered)

    data = data._append(prGm6)
    data = data._append(prFm6)

    data = data._append(prepaid_inductive_0)
    data = data._append(prepaid_inductive_02)
    data = data._append(prepaid_filtered)
    data = data._append(prepaid_max_cov)

    data = data._append(request_inductive_0)
    data = data._append(request_inductive_02)
    data = data._append(request_filtered)
    data = data._append(request_max_cov)

    data = data._append(domestic_inductive_0)
    data = data._append(domestic_inductive_02)
    data = data._append(domestic_filtered)
    data = data._append(domestic_max_cov)

    data = data._append(international_inductive_0)
    data = data._append(international_inductive_02)
    data = data._append(international_filtered)
    data = data._append(international_max_cov)

    data = data._append(permit_inductive_0)
    data = data._append(permit_inductive_02)
    data = data._append(permit_filtered_curr)

    data = data.reset_index(drop=True)

    return data


def create_features(trace, pn):
    curr_features = []

    matches_traces, matches_traces_ratio, matches_models, matches_models_ratio = features.matching_labels(pn, trace)
    curr_features.append(matches_traces)
    curr_features.append(matches_traces_ratio)
    curr_features.append(matches_models)
    curr_features.append(matches_models_ratio)

    len_trace, trace_transitions_ratio, trace_places_ratio, transitions_trace_ratio, places_trace_ratio = features.trace_ratio(
        pn, trace)
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

    free_choice = features.free_choice(pn)
    curr_features.append(free_choice)

    return curr_features


# data = merge_data.get_merged_data()

def label_data(data):
    # Encode data
    # Schnellste Heuristic Time = 1, else 0

    # multi class classification
    # label column
    #plot_data.create_sunburst_plot(data, 180)

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


def get_features_data(data):

    #features_data = []
    feature = []

    start_features = time.time()

    for i in data.index:
        row = data.iloc[i, :]
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

    #features_data.append(feature)

    columns = ["Matching Labels Trace", "Matching Trace Ratio", "Matching Labels Models",
               "Matching Model Ratio", "Trace Length", "Trace Trans Ratio",
               "Trace Place Ratio", "Trans Trace Ratio", "Place Trace Ratio",
               "Trace Loop", "Trace Loop Ratio", "Trace Loop Max",
               "Trace Loop Max Ratio", "Trace Loop Mean", "Trace Loop Mean Ratio",
               "Trace Loop Sum", "Trace Loop Sum Ratio", "Model Duplicates",
               "Model Duplicates Ratio",
               "Trans No In-arc", "Trans No In-arc ratio",
               "Silent Transitions", "Silent Transitions ratio",
               "Parallelism Sum",
               "Parallelism Sum Ratio", "Parallelism Mult", "Parallelism Mult Ratio",
               "Choice Sum", "Choice Sum Ratio", "Choice Mult", "Choice Mult Ratio",
               "Simplicity", "Free-Choice"]

    features_df = pd.DataFrame(feature, columns=columns, dtype=float)

    # scale features
    # scaler = MinMaxScaler()
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_df)
    scaled_df = pd.DataFrame(scaled_features, columns=features_df.columns)

    result = data.join(scaled_df)

    print(result)
    return result, len_features


def get_train_test_set(data, ratio=0.8):
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
        y_train = train["Label"]
        X_test = test.iloc[:, 1:]
        y_test = test["Label"]

        if len(y_train.unique()) == len(y_test.unique()):
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
"""
road_inductive_02 = pd.read_pickle("results/road_inductive_3.pkl")
road_heuristic = pd.read_pickle("results/road_heuristic_3.pkl")
road_filtered = pd.read_pickle("results/road_filtered_inductive_0_3.pkl")

test_road = road_inductive_02.iloc[0:2, :]
test_road = test_road._append(road_heuristic.iloc[0:2, :])
test_road = test_road._append(road_filtered.iloc[0:2, :])
test_road = test_road.reset_index()
data_features, len_features = get_features_data(test_road)
"""
data = get_merged_data()
data_labeled = label_data(data)
data_features, len_features = get_features_data(data_labeled)

# road_heuristic = pd.read_pickle("results/road_heuristic_3.pkl")
# road_features, _ = preprocess_data.get_features_data(road_heuristic)
# road_x_train, road_y_train, _, _ = preprocess_data.get_train_test_set(road_features, 1)

X_train, y_train, X_test, y_test = get_train_test_set(data_features)

# road_x_train = np.asarray(road_x_train.iloc[:, -len_features:], dtype=object)
X_test_features = np.asarray(X_test.iloc[:, -len_features:], dtype=object)
X_train_features = np.asarray(X_train.iloc[:, -len_features:], dtype=object)

#oversample = RandomOverSampler(sampling_strategy='not majority')
#X_over, y_over = oversample.fit_resample(X_train_features, y_train)
#print("Oversampled class distribution:", Counter(y_over))

# X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=0)

"""
# Naive Bayesian
gnb = GaussianNB()

y_pred = gnb.fit(X_train, y_train).predict(X_test)
accuracy = gnb.score(X_test, y_test)
print("Bayesian ", accuracy)
print(
    "Bayesian: Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred).sum()))

# Logistic
model = LogisticRegression(multi_class='ovr', max_iter=1000)
model.fit(X_train, y_train)
y_pred_log = model.predict(X_test)
print("Logistic: Number of mislabeled points out of a total %d points : %d" % (
    X_test.shape[0], (y_test != y_pred_log).sum()))

# SVM

svc = SVC(decision_function_shape='ovo', class_weight="balanced")
svc.fit(X_train_features, y_train)
y_pred_svc = svc.predict(X_test_features)
accuracy = svc.score(X_test_features, y_test)
print("SVC accuracy ", accuracy)
print(
    "SVC: Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred_svc).sum()))
"""
dtree = DecisionTreeClassifier().fit(X_train_features, y_train)
y_pred_tree = dtree.predict(X_test_features)
print("Decision Tree: Number of mislabeled points out of a total %d points : %d" % (
    X_test.shape[0], (y_test != y_pred_tree).sum()))
tree_labels = dtree.classes_

score_tree = metrics.accuracy_score(y_test, y_pred_tree)
print("Tree accuracy when no difference between LP and ILP:   %0.3f" % score_tree)

#proba_tree = dtree.predict_proba(X_test_features)

cm = metrics.confusion_matrix(y_test, y_pred_tree, labels=tree_labels)

df_cm = pd.DataFrame(cm, columns=tree_labels, index=tree_labels)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16},fmt='g')
plt.title("Confusions Matrix of Decision Tree")
plt.show()

# NN

start_train_mlp = time.time()

mlp = MLPClassifier(solver='adam', hidden_layer_sizes=(len_features, len_features, 10, 4), learning_rate_init=0.001, random_state=1,
                    max_iter=1000)
mlp.out_activation_ = 'softmax'
# mlp.fit(road_x_train, road_y_train)
mlp.fit(X_train_features, y_train)
#mlp.fit(X_over, y_over)

end_train_mlp = time.time()
time_train_mlp = end_train_mlp - start_train_mlp

start_predict_mlp = time.time()
y_pred_mlp = mlp.predict(X_test_features)

end_predict_mlp = time.time()
time_predict_mlp = end_predict_mlp - start_predict_mlp

proba_mlp = mlp.predict_log_proba(X_test_features)

print("MLP: Number of mislabeled points out of a total %d points : %d" % (
    X_test.shape[0], (y_test != y_pred_mlp).sum()))

score = metrics.accuracy_score(y_test, y_pred_mlp)
print("MLP accuracy:   %0.3f" % score)

mlp_labels = mlp.classes_

class_report = metrics.classification_report(y_test, y_pred_mlp, target_names=mlp_labels)

cm = metrics.confusion_matrix(y_test, y_pred_mlp, labels=mlp_labels)

df_cm = pd.DataFrame(cm, columns=mlp_labels, index = mlp_labels)
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16},fmt='g')
plt.title("Confusions Matrix of Neural Network")
#sns.heatmap(cm, annot=mlp_labels, cmap='Blues')
plt.show()
#disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mlp_labels)
#disp.plot()
#plt.show()

# Ensembles

# print("Time to compute features: ", time_features)
print("Time to train mlp: ", time_train_mlp)
print("Time to predict with mlp: ", time_predict_mlp)

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
print("MLP accuracy when no difference between LP and ILP:   %0.3f" % score_relax)

optimal_time = statistical_numbers.time_with_optimal_heuristics(X_test)
time_model = statistical_numbers.time_using_model(X_test, y_pred_mlp)
time_heuristics = statistical_numbers.time_using_one_heuristic(X_test)

optimal_lps = statistical_numbers.lps_optimal_heuristics(X_test)
lps_one_heuristic = statistical_numbers.lps_using_one_heuristic(X_test)

# timeouts
timeout = 180
optimal_timeouts, _ = statistical_numbers.timeouts_optimal_heuristics(X_test, timeout)

timeouts_heuristics = []

heuristics = ["No Heuristic Time", "Naive Time", "State Eq. LP Time", "State Eq. ILP Time", "Ext. Eq. LP Time",
              "Ext. Eq. ILP Time"]

for h in heuristics:
    num_timeouts = len(X_test[X_test[h].ge(timeout)])
    timeouts_heuristics.append(num_timeouts)

# states
optimal_visited_states = statistical_numbers.states_optimal_heuristics(X_test, True)
optimal_queued_states = statistical_numbers.states_optimal_heuristics(X_test, False)

visited_states_heuristic = statistical_numbers.states_one_heuristic(X_test, True)
queued_states_heuristic = statistical_numbers.states_one_heuristic(X_test, False)

time_relaxed_model = statistical_numbers.time_using_model(X_test, y_pred_relax_lp)

statistical_numbers.plot_multiple_bars(optimal_time, time_model, time_heuristics, "Time in seconds", "Computation Time")

# ml model
y_pred_mlp_ls = y_pred_mlp.tolist()
model_idx = statistical_numbers.replace_label_by_time(y_pred_mlp.tolist())
model_timeouts, _ = statistical_numbers.timeouts_optimal_heuristics(X_test, timeout, model_idx)
model_lps = statistical_numbers.lps_optimal_heuristics(X_test, model_idx)
model_visited_states = statistical_numbers.states_optimal_heuristics(X_test, True, model_idx)
model_queued_states = statistical_numbers.states_optimal_heuristics(X_test, False, model_idx)

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

plt.rcParams.update(plt.rcParamsDefault)
"""
statistical_numbers.plot_multiple_bars_h(optimal_time, [random_time, time_model], time_heuristics,
                                       "Time in seconds", "Computation Time MLP Model")
statistical_numbers.plot_multiple_bars_h(optimal_queued_states, [random_queued_states, model_queued_states],
                                       queued_states_heuristic, "Number of queued states",
                                       "Queued States")

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

