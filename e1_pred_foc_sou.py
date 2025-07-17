from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import random
import numpy as np

from egci_bioacoustic_shifts import load_EGCI


# TODO: Move data into its own file
def get_X_Y(data, reshuffle_labels=False):
    labels = data[:, 2]
    if reshuffle_labels:
        random.shuffle(labels)
    data[:, 2] = labels
    return data

def run_trial(dataset, get_X_Y, reshuffle_labels=False, test_size=0.2):
    X, y = get_X_Y(dataset, reshuffle_labels=reshuffle_labels)
    print(X.shape, y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    clf = SVC()
    clf.fit(X_train, y_train)
    return clf.score(X_test, y_test)

def run_experiment(dataset_info, get_X_Y, iterations=100):
    sample_stat = run_trial(dataset_info, get_X_Y, reshuffle_labels=False)

    null_hypo_trial_results = []
    for i in range(iterations):
        null_hypo_trial_results.append(run_trial(dataset_info, get_X_Y,  reshuffle_labels=True))

    return {
        "sample_stat": sample_stat,
        "p-value": (sample_stat < np.array(null_hypo_trial_results)).mean(),
        "null_hypo_trials": null_hypo_trial_results
    }


# Experiment Parameters
regions = ["HSN", "PER", "UHH", "SNE", "POW", "COR"]
num_samples = 2000
num_trials = 100

# Actual Experiment
experiment_results = {}
for region in regions:
    _, _, soundscape_data, _ = load_EGCI(indx=num_samples, region=region, dataset_sub="test_5s")
    _, _, focal_data, _ = load_EGCI(indx=num_samples, region=region, dataset_sub="train")
    
    # Format focal and soundscape EGCI for SVM
    labels = []
    (h, c, ) = soundscape_data
    labels.extend([0] * num_samples)

    h.extend(focal_data[0])
    c.extend(focal_data[1])
    labels.extend([1] * num_samples)

    data = np.vstack((np.array(h), np.array(c)), np.array(labels)).T
    np.random.shuffle(data)

    # Run SVM training and permutation test
    experiment_results[region] = run_experiment(data, get_X_Y, iterations=num_trials)
    print(experiment_results[region] )

    # TODO Add in the normal divergence metrics


