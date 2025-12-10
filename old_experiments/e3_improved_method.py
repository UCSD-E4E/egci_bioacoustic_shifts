import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TORCH_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import statsmodels.api as sm
import pandas as pd
import numpy as np

from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments, extractors
from pyha_analyzer.models.demo_CNN import ResnetConfig, ResnetModel
from pyha_analyzer.preprocessors import MelSpectrogramPreprocessors, MixItUp, ComposeAudioLabel
from audiomentations import Compose, AddColorNoise, AddBackgroundNoise, PolarityInversion, Gain
from pyha_analyzer.models import EfficentNet

import numpy as np
import torch
import random
import json
import librosa
from tqdm import tqdm


torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

import torch

torch.set_num_threads(1)

#   export OMP_NUM_THREADS=4
#     export MKL_NUM_THREADS=4
#     export TORCH_NUM_THREADS=4


import warnings
warnings.filterwarnings("ignore") #AUDIOMENTIONS REALLY NEEDS TO QUIET RESAMPLING WARNINGS

from egci_bioacoustic_shifts import EGCI, process_data, measure_distrbution_metrics

def load_file(path):
    y, sr  = librosa.load(path=path, sr=32_000)
    return y

def combine_files(y_1, y_2):
    if y_1.shape[-1] > y_2.shape[-1]:
        start = np.random.randint(0, y_1.shape[-1] - y_2.shape[-1])
        y_1 = y_1[start:start + y_2.shape[-1]]
    if y_2.shape[-1] > y_1.shape[-1]:
        start = np.random.randint(0, y_2.shape[-1] - y_1.shape[-1])
        y_2 = y_2[start:start + y_1.shape[-1]]

    return y_1, y_2

def get_data_per_region(ds):
    data = []

    for i in tqdm(range(len(ds["valid"]))):

        vaild_sample = ds["valid"][int(i)]
        test_sample_idx = np.random.choice(np.where(np.array(ds["test"]["labels"])[:, vaild_sample["ebird_code"]] == 1)[0])
        test_sample = ds["test"][int(test_sample_idx)]

        valid_y = load_file(vaild_sample["audio"]["path"])
        test_y = load_file(test_sample["audio"]["path"])

        test_weight = 0.5
        test_y, valid_y = combine_files(test_y, valid_y)
        data.append((valid_y.tolist(), test_y.tolist(), vaild_sample, test_sample))

    return data


import torchvision.transforms as transforms

def fake_preprocessor(weighted_y, vaild_sample):
    n_fft=2048
    hop_length=256
    power=2.0
    n_mels=256

    pillow_transforms = transforms.ToPILImage()
        
    mels = np.array(
        pillow_transforms(
            librosa.feature.melspectrogram(
                y=weighted_y, sr=32_000,
                n_fft=n_fft, 
                hop_length=hop_length, 
                power=power, 
                n_mels=n_mels, 
            )
        ),
        np.float32)[np.newaxis, ::] / 255

    # model()
    batch = {
        "audio_in": torch.Tensor([mels]).cuda(),
        "audio": torch.Tensor([mels]).cpu(),
        "labels": torch.Tensor([vaild_sample["labels"]]).cuda()
    }
    return batch

def compute_egci(data, model, preprocessor):
    H = defaultdict(list)
    C = defaultdict(list)
    H_avgs = []
    C_avgs = []
    outs = defaultdict(list)
    total = 10
    for i in tqdm(range(total + 1),  position=0, leave=True):
        for (valid_y, test_y, vaild_sample, test_sample) in tqdm(data,  position=1, leave=False):
            weight = i / total
            weighted_y = (1 - weight) * np.array(valid_y) + weight * np.array(test_y)
            
            ## TODO FORMAT MODEL INPUT
            out = model(**fake_preprocessor(weighted_y, vaild_sample))
            
            h,c,lag = EGCI(weighted_y, 256)
            H[i].append(float(h))
            C[i].append(float(c))
            outs[i].append({
                "loss": out["loss"].detach().cpu().numpy().tolist(),
                "logits": out["logits"].detach().cpu().numpy().tolist(),
            })

        H_avgs.append(float(np.mean(H[i])))
        C_avgs.append(float(np.mean(C[i])))
        print(H_avgs[-1], C_avgs[-1])

    return H,C,outs

# Experiment Parameters
regions = ["HSN", "PER", "UHH", "SNE", "POW", "NES"]
num_samples = 2000
num_trials = 100
birdset_extactor = extractors.Birdset()


# GAME PLAN

# For each region
#   Train a model over that region, no augmentation
#   Sample annotations based on e3_test_augmented_pair.ipynb
#   For each "augment" pair
#       do a weighted average of the two clips
#       Run movel over the weighted pair, go based on focal species score?
#       Get the EGCI of that weighted pair
#   Compute region metrics here? Defeintly save all raw outputs


experiment_results = {}
experiment_results_tiny = {}
for region in regions:
    ds = birdset_extactor(region)
    ## STEP ONE TRAIN MODEL
    experiment_results[region] = {}
    experiment_results_tiny[region] = {}


    preprocessor = MelSpectrogramPreprocessors(
        duration=5, 
        augment=None,
    )

    ds["train"].set_transform(preprocessor)
    ds["valid"].set_transform(preprocessor)
    ds["test"].set_transform(preprocessor)

    model = EfficentNet(num_classes=len(ds["train"].features["ebird_code"].names))

    args = PyhaTrainingArguments(
        working_dir="working_dir"
    )
    args.num_train_epochs = 50
    args.eval_steps = 200
    args.dataloader_num_workers = 16
    args.per_device_train_batch_size = 24
    args.per_device_eval_batch_size = 24
    args.learning_rate = 0.001
    
    args.run_name = "e3_new_method_" + region

    trainer = PyhaTrainer(
        model=model,
        dataset=ds,
        training_args=args
    )
    #trainer.evaluate(eval_dataset=hsn_ads["test"], metric_key_prefix="Soundscape_test")
    
    # TODO TESTING
    trainer.train()


    ## STEP TWO: GET TEST SAMPLES
    # ds = birdset_extactor(region) #Reset data preprocessors
    ds.reset_format()
    data = get_data_per_region(ds)

    ## STEP THREE: RUN MODEL OVER "AUGMENTED"
    ## IF DOMAIN SHIFT HOLDS, EGCI AND MODEL PERFORMANCE WILL CHANGE
    H,C,outs = compute_egci(data, model, preprocessor)

    experiment_results[region] = {
        # "data": data,
        "H": H,
        "C": C,
        "outs": outs
    }

    del data

    # experiment_results_tiny[region] = {
    #     "H": H,
    #     "C": C,
    #     "outs": outs
    # }

    ## STEP FOUR:
    ## SAVE RESULTS FOR STUDY

    with open("e3_results_new_method_temp.json", "w") as file:
        json.dump(experiment_results, file, indent=4)
    
    # with open("e3_results_new_method_tiny.json", "w") as file:
    #     json.dump(experiment_results_tiny, file, indent=4)