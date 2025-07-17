from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import statsmodels.api as sm
import pandas as pd
import numpy as np

from utils import load_EGCI

# Experiment Parameters
regions = ["HSN", "PER", "UHH", "SNE", "POW", "COR"]
num_samples = 2000
num_trials = 100

# Actual Experiment
experiment_results = {}
for region in regions:
    pass
    # Compute EGCI stats with and without data augmentations

    # Compute Model performance with and without augmentations

    # Save results of both experiments