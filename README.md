# EGCI Bioacoustic Shifts

## Install

To install dependencies, make sure you have `uv` on your machine and then run.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git submodule update --init --recursive
uv sync --extra cu126
```

## How to Use
There are two parts to every experiment, a python script for preprocessing indicies and a notebook for studying the indicies output
Data is automatically downloaded as the scripts are run, see https://huggingface.co/datasets/DBD-research-group/BirdSet for details

In both notebooks and python script there is a boolean parameter `only_1_label_each` that swaps between the base mode of running the data over all soundscapes if False or the alterative mode that compares only soundscapes of a single label example. 

### Experiment 1
1) To get data and indices, run `uv run e1_pred_foc_sou.py`
2) Access e1_better_stats.ipynb and e1_ttest.ipynb from a prefered jupyter notebook renderer for the statistical testing done in e1

### Experiment 2
1) To get data, indices, and model predictions, run `uv run e2_linear_reg_loss.py`
2) Access e2_better_stats.ipynb from a prefered jupyter notebook renderer for the statistical testing done in e2
