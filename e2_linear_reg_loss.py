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
    fig, out, (h, c, preds, losses, labels),  indx = load_EGCI(indx=num_samples, region=region, dataset_sub="test_5")
    # fig, out, focal_data,  indx = load_EGCI(indx=num_samples, region=region, dataset_sub="train")
    
    df = pd.DataFrame({"Entropy": h, "Complexity": c, "Loss": losses, "GT": labels, "Predictions": preds})
    X = sm.add_constant(df[['Entropy', 'Complexity', 'GT']])
    model = sm.OLS(df['Loss'], X).fit()

    experiment_results[region] = model.summary()