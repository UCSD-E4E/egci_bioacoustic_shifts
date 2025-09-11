from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import random
import numpy as np
import copy
import json
from egci_bioacoustic_shifts import load_EGCI, measure_distrbution_metrics

# Experiment Parameters
regions = ["HSN", "PER", "UHH", "SNE", "POW", "NES"]
num_samples = 2000
num_trials = 1000


# TODO: Move data into its own file
def get_X_Y(data, reshuffle_labels=False):
    labels = data[:, 2]
    if reshuffle_labels:
        random.shuffle(labels)
    data[:, 2] = labels
    return data[:, :2], labels

def run_trial(dataset, get_X_Y, reshuffle_labels=False, test_size=0.2):
    X, y = get_X_Y(dataset, reshuffle_labels=reshuffle_labels)
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    clf = SVC()
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test), clf, X_test, y_test

# def run_experiment(dataset_info, get_X_Y, iterations=100):
#     sample_stat = run_trial(dataset_info, get_X_Y, reshuffle_labels=False)

#     null_hypo_trial_results = []
#     for i in range(iterations):
#         null_hypo_trial_results.append(run_trial(dataset_info, get_X_Y,  reshuffle_labels=True))

#     return {
#         "sample_stat": sample_stat,
#         "p-value": (sample_stat < np.array(null_hypo_trial_results)).mean(),
#         "null_hypo_trials": null_hypo_trial_results
#     }


# Actual Experiment
experiment_results = {}
for region in regions:              #Change sample to indx
    _, _, soundscape_data, _ = load_EGCI(sample=num_samples, region=region, dataset_sub="test_5s")
    _, _, focal_data, _ = load_EGCI(sample=num_samples, region=region, dataset_sub="train")
    
    # Format focal and soundscape EGCI for SVM
    labels = []
    (h, c, _) = copy.deepcopy(soundscape_data)
    labels.extend([0] * len(h))

    h.extend(focal_data[0])
    c.extend(focal_data[1])
    labels.extend([1] * len(focal_data[0]))

    data = np.vstack((np.array(h), np.array(c), np.array(labels))).T
    np.random.shuffle(data)
    print(data[0], data.shape)

    # Run SVM training and permutation test
    experiment_results[region] = {}

    sample_stat, clf, X_test, y_test = run_trial(data, get_X_Y, reshuffle_labels=False, test_size=0.2)


    scores = []
    for i in range(num_trials):
        np.random.shuffle(y_test)
        scores.append(clf.score(X_test, y_test))
    
    experiment_results[region]["svm"] = {
        "sample_stat": sample_stat,
        "p-value": (sample_stat < np.array(scores)).mean(),
        "null_hypo_trials": scores
    }

    # Removed to test permutations on model output for better test
    # experiment_results[region]["svm"] = run_experiment(data, get_X_Y, iterations=num_trials)
    

    # TODO Add in the normal divergence metrics
    print("start divergence")
    experiment_results[region]["div"] = measure_distrbution_metrics(
        np.vstack((np.array(focal_data[0]), np.array(focal_data[1]))).T,
        np.vstack((np.array(soundscape_data[0]), np.array(soundscape_data[1]))).T,
    )
    print(experiment_results[region])

    experiment_results[region]["data"]  = {
        "soundscape": soundscape_data,
        "focal": focal_data,
    }

with open("e1_results.json", "w") as file:
    json.dump(experiment_results, file, indent=4)
