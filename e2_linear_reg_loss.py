from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import statsmodels.api as sm
import statsmodels.stats.api as sms
import pandas as pd
import numpy as np
import json
from statsmodels.compat import lzip

from egci_bioacoustic_shifts import load_EGCI_losses

# Experiment Parameters
regions = ["HSN", "PER", "UHH", "SNE", "POW", "NES"]
num_samples = 2000
num_trials = 100

# Actual Experiment
experiment_results = {}
for region in regions:

    fig, out, (h, c, preds, losses, labels),  indx = load_EGCI_losses(indx=num_samples, region=region, dataset_sub="test_5s")
    # fig, out, focal_data,  indx = load_EGCI(indx=num_samples, region=region, dataset_sub="train")
    
    df = pd.DataFrame({"Entropy": h, "Complexity": c, "Loss": losses, "GT": labels, "Predictions": preds})
    df["GT"] = df["GT"].apply(sum)
    X = sm.add_constant(df[['Entropy', 'Complexity', 'GT']])
    model = sm.OLS(df['Loss'], X).fit()

    name = ["t value", "p value"]
    test = sms.linear_harvey_collier(model)

    experiment_results[region] = {}
    experiment_results[region]["results"] = (str(model.summary()), str(lzip(name, test)))
    print(experiment_results[region])
    experiment_results[region]["data"]  = {
        "soundscape": df.to_json()
    }


print(experiment_results)
with open("e2_results.json", "w") as file:
    json.dump(experiment_results, file, indent=4)
