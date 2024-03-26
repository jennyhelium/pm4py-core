import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

import features


def create_features(trace, pn):
    curr_features = []

    matches_traces, matches_traces_ratio, matches_models, matches_models_ratio = features.matching_labels(pn, trace)
    curr_features.append(matches_traces)
    curr_features.append(matches_traces_ratio)
    curr_features.append(matches_models)
    curr_features.append(matches_models_ratio)

    len_trace, trace_transitions_ratio, trace_places_ratio = features.trace_ratio(pn, trace)
    curr_features.append(len_trace)
    curr_features.append(trace_transitions_ratio)
    curr_features.append(trace_places_ratio)

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

    choice_sum, choice_ratio = features.choice(pn)
    curr_features.append(choice_sum)
    curr_features.append(choice_ratio)

    return curr_features


data = pd.read_pickle("results/domestic_inductive_3.pkl")

data2 = pd.read_pickle("results/italian_alpha_0.2_3.pkl")

data = data._append(data2)

# Encode data
# Schnellste Heuristic Time = 1, else 0

# multi class classification
# label column

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
            labels.append("State Eq. ILP")
        elif i == "Ext. Eq. LP Time":
            labels.append("Ext. Eq. LP")
        elif i == "Ext. Eq. ILP Time":
            labels.append("Ext. Eq. ILP")

    row_num = row_num + 1

encode_times.insert(0, "Label", labels)

# one_hot_encoded = pd.get_dummies(encode_times, columns=["Label"])

features_data = []
feature = []

for i in range(len(encode_times)):
    row = encode_times.iloc[i, :]
    trace = row["Trace"]
    pn = row["Petri Net"]
    im = row["Initial Marking"]
    fm = row["Final Marking"]

    row_features = create_features(trace, pn)

    len_features = len(row_features)

    feature.append(row_features)

features_data.append(feature)

features_df = pd.DataFrame(feature,
                           columns=["Matching Labels Trace", "Matching Trace Ratio", "Matching Labels Models",
                                    "Matching Model Ratio", "Trace Length", "Trace Trans Ratio", "Trace Place Ratio",
                                    "Trace Loop", "Trace Loop Ratio", "Trace Loop Max",
                                    "Trace Loop Max Ratio", "Trace Loop Mean", "Trace Loop Mean Ratio",
                                    "Trace Loop Sum", "Trace Loop Sum Ratio", "Model Duplicates",
                                    "Model Duplicates Ratio", "Trans No In-arc", "Trans No In-arc ratio",
                                    "Silent Transitions", "Silent Transitions ratio", "Parallelism Sum",
                                    "Parallelism Sum Ratio", "Choice Sum", "Choice Sum Ratio"])

# scale features
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(features_df)
scaled_df = pd.DataFrame(scaled_features, columns=features_df.columns)

result = encode_times.join(scaled_df)

print(result)

X = np.asarray(result.iloc[:, -len_features:], dtype=object)
# y = np.asarray(result.iloc[:, :1], dtype=object)
y = result["Label"]

oversample = RandomOverSampler(sampling_strategy='minority')
X_over, y_over = oversample.fit_resample(X, y)
print("Oversampled class distribution:", Counter(y_over))

X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.2, random_state=0)

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
svc = SVC(decision_function_shape='ovo')
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
accuracy = svc.score(X_test, y_test)
print("SVC accuracy ", accuracy)
print(
    "SVC: Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred_log).sum()))

dtree = DecisionTreeClassifier().fit(X_train, y_train)
y_pred_tree = dtree.predict(X_test)
print("Decision Tree: Number of mislabeled points out of a total %d points : %d" % (
    X_test.shape[0], (y_test != y_pred_log).sum()))

# NN

mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(6,), random_state=1)
mlp.out_activation_ = 'softmax'
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
print("MLP: Number of mislabeled points out of a total %d points : %d" % (
    X_test.shape[0], (y_test != y_pred_mlp).sum()))

# Ensembles
