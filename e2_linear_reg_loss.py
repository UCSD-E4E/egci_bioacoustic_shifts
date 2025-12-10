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
regions = ["SSW"] #["PER", "UHH", "SNE", "POW", "NES"] # "HSN"
num_samples = 2000
num_trials = 100
name = ["t value", "p value"]
# Actual Experiment
experiment_results = {}
for region in regions:
    experiment_results[region] = {}
    for i in range(num_trials):
        experiment_results[region][i] = {}

        fig, out, (h, c, preds, losses, labels), indx, outs = load_EGCI_losses(indx=num_samples, region=region, dataset_sub="test_5s")
        # fig, out, focal_data,  indx = load_EGCI(indx=num_samples, region=region, dataset_sub="train")
        
        df = pd.DataFrame({"Entropy": h, "Complexity": c, "Loss": losses, "GT": labels, "Predictions": preds})
        df["GT"] = df["GT"].apply(sum)
        # X = sm.add_constant(df[['Entropy', 'Complexity', 'GT']])
        # ols_model = sm.OLS(df['Loss'], X).fit()
        # ols_test = sms.linear_harvey_collier(ols_model)

        # Huber_model = sm.RLM(df['Loss'], X, M=sm.robust.norms.HuberT()).fit()
        # Huber_test = sms.linear_harvey_collier(Huber_model)

        
        # experiment_results[region][i]["ols_results"] = (str(ols_model.summary()), str(lzip(name, ols_test)))
        # experiment_results[region][i]["Huber_results"] = (str(Huber_model.summary()), str(lzip(name, Huber_test)))

        # print(experiment_results[region][i])
        experiment_results[region][i]["data"]  = {
            "soundscape": df.to_json(),
            "raw_data": outs
        }

        print(experiment_results)
        with open("e2_results_SSW.json", "w") as file:
            json.dump(experiment_results, file, indent=4)
    print(experiment_results)
    with open("e2_results_SSW.json", "w") as file:
        json.dump(experiment_results, file, indent=4)



##### Revist with more metrics and Better Anasysis

