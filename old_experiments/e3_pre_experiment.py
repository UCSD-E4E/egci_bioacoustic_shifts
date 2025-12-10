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

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

import warnings
warnings.filterwarnings("ignore") #AUDIOMENTIONS REALLY NEEDS TO QUIET RESAMPLING WARNINGS

from egci_bioacoustic_shifts import load_EGCI, process_data, measure_distrbution_metrics

# Experiment Parameters
regions = ["HSN", "PER", "UHH", "SNE", "POW", "NES"]
num_samples = 2000
num_trials = 100
birdset_extactor = extractors.Birdset()

# Augmentation function
class AugmentAudio():
    def __init__(self, augmentations):
        self.augmentations = augmentations
    
    def __call__(self, data):
        if self.augmentations is not None:
            return process_data(data, audio_processing=self.augmentations)
        return process_data(data)

experiment_results = {}
for region in regions:

    _, _, soundscape_data, _ = load_EGCI(sample=num_samples, region=region, dataset_sub="test_5s")

    ads = birdset_extactor(region=region)
    experiment_parameters = [
        {
            "augmentation": ComposeAudioLabel([
                AddBackgroundNoise(
                    sounds_path="wav",
                    min_snr_db=-40,
                    max_snr_db=40,
                    noise_transform=PolarityInversion(),
                    p=0.5
                ),
                Gain(
                    min_gain_db = -40,
                    max_gain_db = 40,
                    p = 0.2
                ),
                MixItUp(
                    dataset_ref=ads["train"],
                    min_snr_db=-40,
                    max_snr_db=40,
                    noise_transform=PolarityInversion(),
                    p=0.7
                )

            ]),
            "run_name": "aug8",
            "region": region
        },
        {
            "augmentation": ComposeAudioLabel([
                AddBackgroundNoise(
                    sounds_path="wav",
                    min_snr_db=-20,
                    max_snr_db=10,
                    noise_transform=PolarityInversion(),
                    p=0.5
                ),
                Gain(
                    min_gain_db = -20,
                    max_gain_db = 10,
                    p = 0.2
                ),
                MixItUp(
                    dataset_ref=ads["train"],
                    min_snr_db=-20,
                    max_snr_db=10,
                    noise_transform=PolarityInversion(),
                    p=0.7
                )

            ]),
            "run_name": "aug7",
            "region": region
        },
        {
            "augmentation": ComposeAudioLabel([
                AddBackgroundNoise(
                    sounds_path="wav",
                    min_snr_db=-30,
                    max_snr_db=10,
                    noise_transform=PolarityInversion(),
                    p=0.5
                ),
                Gain(
                    min_gain_db = -30,
                    max_gain_db = 10,
                    p = 0.2
                ),
                MixItUp(
                    dataset_ref=ads["train"],
                    min_snr_db=-30,
                    max_snr_db=10,
                    noise_transform=PolarityInversion(),
                    p=0.7
                )

            ]),
            "run_name": "aug6",
            "region": region
        },
        {
            "augmentation": ComposeAudioLabel([
                AddBackgroundNoise(
                    sounds_path="wav",
                    min_snr_db=-10,
                    max_snr_db=20,
                    noise_transform=PolarityInversion(),
                    p=0.5
                ),
                Gain(
                    min_gain_db = -10,
                    max_gain_db = 20,
                    p = 0.2
                ),
                MixItUp(
                    dataset_ref=ads["train"],
                    min_snr_db=-10,
                    max_snr_db=20,
                    noise_transform=PolarityInversion(),
                    p=0.7
                )

            ]),
            "run_name": "aug5",
            "region": region
        },
        {
            "augmentation": ComposeAudioLabel([
                AddBackgroundNoise(
                    sounds_path="wav",
                    min_snr_db=-10,
                    max_snr_db=30,
                    noise_transform=PolarityInversion(),
                    p=0.5
                ),
                Gain(
                    min_gain_db = -10,
                    max_gain_db = 30,
                    p = 0.2
                ),
                MixItUp(
                    dataset_ref=ads["train"],
                    min_snr_db=-10,
                    max_snr_db=30,
                    noise_transform=PolarityInversion(),
                    p=0.7
                )

            ]),
            "run_name": "aug4",
            "region": region
        },
        {
            "augmentation": ComposeAudioLabel([
                AddBackgroundNoise(
                    sounds_path="wav",
                    min_snr_db=-20,
                    max_snr_db=20,
                    noise_transform=PolarityInversion(),
                    p=0.5
                ),
                Gain(
                    min_gain_db = -20,
                    max_gain_db = 20,
                    p = 0.2
                ),
                MixItUp(
                    dataset_ref=ads["train"],
                    min_snr_db=-20,
                    max_snr_db=20,
                    noise_transform=PolarityInversion(),
                    p=0.7
                )

            ]),
            "run_name": "aug3",
            "region": region
        },
        {
            "augmentation": ComposeAudioLabel([
                AddBackgroundNoise(
                    sounds_path="wav",
                    min_snr_db=-30,
                    max_snr_db=30,
                    noise_transform=PolarityInversion(),
                    p=0.5
                ),
                Gain(
                    min_gain_db = -30,
                    max_gain_db = 30,
                    p = 0.2
                ),
                MixItUp(
                    dataset_ref=ads["train"],
                    min_snr_db=-30,
                    max_snr_db=30,
                    noise_transform=PolarityInversion(),
                    p=0.7
                )

            ]),
            "run_name": "aug2",
            "region": region
        },
        {
            "augmentation": ComposeAudioLabel([
                AddBackgroundNoise(
                    sounds_path="wav",
                    min_snr_db=-10,
                    max_snr_db=10,
                    noise_transform=PolarityInversion(),
                    p=0.5
                ),
                Gain(
                    min_gain_db = -10,
                    max_gain_db = 10,
                    p = 0.2
                ),
                MixItUp(
                    dataset_ref=ads["train"],
                    min_snr_db=-10,
                    max_snr_db=10,
                    noise_transform=PolarityInversion(),
                    p=0.7
                )

            ]),
            "run_name": "aug1",
            "region": region
        }, 
        {
            "augmentation": ComposeAudioLabel([
                AddBackgroundNoise(
                    sounds_path="wav",
                    min_snr_db=-5,
                    max_snr_db=5,
                    noise_transform=PolarityInversion(),
                    p=0.5
                ),
                Gain(
                    min_gain_db = -5,
                    max_gain_db = 5,
                    p = 0.2
                ),
                MixItUp(
                    dataset_ref=ads["train"],
                    min_snr_db=-5,
                    max_snr_db=5,
                    noise_transform=PolarityInversion(),
                    p=0.7
                )

            ]),
            "run_name": "aug9",
            "region": region
        }, 
        {
            "augmentation": None,
            "run_name": "base",
            "region": region
        }
    ]
    experiment_results[region] = {}

    focal_samples = num_samples
    for parameters in experiment_parameters:
        experiment_results[region][parameters["run_name"]] = {}


        preprocessor = MelSpectrogramPreprocessors(
            duration=5, 
            augment=parameters["augmentation"],
        )

        test_preprocessor = MelSpectrogramPreprocessors(
            duration=5, 
            augment=None,
        )

        ads["train"].set_transform(preprocessor)
        ads["valid"].set_transform(test_preprocessor)
        ads["test"].set_transform(test_preprocessor)

        model = EfficentNet(num_classes=len(ads["train"].features["ebird_code"].names))

        args = PyhaTrainingArguments(
            working_dir="working_dir"
        )
        args.num_train_epochs = 3
        args.eval_steps = 1000
        args.dataloader_num_workers = 20
        args.per_device_train_batch_size = 8
        args.per_device_eval_batch_size = 8
        args.learning_rate = 0.001
        
        args.run_name = parameters["run_name"] + " " + parameters["region"]

        trainer = PyhaTrainer(
            model=model,
            dataset=ads,
            training_args=args
        )
        #trainer.evaluate(eval_dataset=hsn_ads["test"], metric_key_prefix="Soundscape_test")
        
        
        trainer.train()
        experiment_results[parameters["region"]][parameters["run_name"]] = trainer.evaluate(eval_dataset=ads["test"], metric_key_prefix="Soundscape")

        del model
        del trainer

        # _, _, focal_data, focal_samples = load_EGCI(sample=num_samples, region=region, dataset_sub="train")
        process_aug = AugmentAudio(parameters["augmentation"])
        _, _, aug_focal_data, focal_samples = load_EGCI(
            process_data_func=process_aug ,
            sample=focal_samples, region=region, dataset_sub="train")
        
        # # Format focal and soundscape EGCI for S
        # experiment_results[region]["div"] = {}
        # experiment_results[region]["div"]["no_aug"] = measure_distrbution_metrics(
        #     np.vstack((np.array(focal_data[0]), np.array(focal_data[1]))).T,
        #     np.vstack((np.array(soundscape_data[0]), np.array(soundscape_data[1]))).T,
        # )
        # experiment_results[region]["div"]["aug"] = measure_distrbution_metrics(
        #     np.vstack((np.array(aug_focal_data[0]), np.array(aug_focal_data[1]))).T,
        #     np.vstack((np.array(soundscape_data[0]), np.array(soundscape_data[1]))).T,
        # )

        experiment_results[parameters["region"]][parameters["run_name"]]["data"]  = {
            "soundscape": soundscape_data,
            # "focal": focal_data,
            "focal": aug_focal_data,
            "augmented": parameters["augmentation"] is not None
        }
        # Compute EGCI stats with and without data augmentations
        # Save results of both experiments
        with open("e3_results_data_aug_search.json", "w") as file:
            json.dump(experiment_results, file, indent=4)