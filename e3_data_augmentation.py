from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import statsmodels.api as sm
import pandas as pd
import numpy as np

from pyha_analyzer import PyhaTrainer, PyhaTrainingArguments, extractors, AudioDataset
from pyha_analyzer.models.demo_CNN import ResnetConfig, ResnetModel
from pyha_analyzer.preprocessors import MelSpectrogramPreprocessors, MixItUp, ComposeAudioLabel, MelSpectrogramPreprocessorsNew
from audiomentations import Compose, AddColorNoise, AddBackgroundNoise, PolarityInversion, Gain
from pyha_analyzer.models import EfficentNet

from load_audio_birdset import load_audio


import numpy as np
import torch
import random
import json
import math
import os

from datasets import load_from_disk, Audio

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

import warnings
warnings.filterwarnings("ignore") #AUDIOMENTIONS REALLY NEEDS TO QUIET RESAMPLING WARNINGS

from egci_bioacoustic_shifts import load_EGCI, process_data, measure_distrbution_metrics

# Experiment Parameters
regions = ["PER"]
num_samples = 2000
num_trials = 100
birdset_extactor = extractors.Birdset()

# Augmentation function
class AugmentAudio():
    def __init__(self, augmentations):
        self.augmentations = augmentations
    
    def __call__(self, data):
        if self.augmentations:
            return process_data(data, audio_processing=self.augmentations)
        return process_data(data)


print('here before experiment')
experiment_results = {}
for region in regions:
    print('here')
    ds = load_from_disk('/home/s.dalal.334/BirdSet/data_birdset/HSN/HSN_processed_42_e6047e15bfe4c847_b8f572cbedd4294a_aa6b74949732368a')


    print('got past extractor')
    experiment_parameters = [
        # {
        #     "augmentation": ComposeAudioLabel([
        #         AddBackgroundNoise(
        #             sounds_path="data_birdset/background_noise",
        #             min_snr_db=-10,
        #             max_snr_db=10,
        #             noise_transform=PolarityInversion(),
        #             p=0.5
        #         ),
        #         Gain(
        #             min_gain_db = -10,
        #             max_gain_db = 10,
        #             p = 0.2
        #         ),
        #         MixItUp(
        #             dataset_ref=ads["train"],
        #             min_snr_db=-10,
        #             max_snr_db=10,
        #             noise_transform=PolarityInversion(),
        #             p=0.7
        #         )

        #     ]),
        #     "run_name": "birdset_augmentations_Background-Gain-MixitUp",
        #     "region": region
        # },
        {
            "augmentation": None,
            "run_name": "Base, weight_decay+scheduler+warmup",
            "region": region
        }
    ]
    experiment_results[region] = {}

    for parameters in experiment_parameters:

        preprocessor = MelSpectrogramPreprocessors(
            duration=5,
            augment=None,
        )

        test_preprocessor = MelSpectrogramPreprocessors(
            duration=5, 
            augment=None,
        )

        print('here', ds)

        print(ds["train"][0])

        base_storage = '/home/s.dalal.334/BirdSet'
            

        def process_audio(example):
            raw_path = str(example['filepath']) 
            example['filepath'] = raw_path.replace('../..', base_storage)
            
            # 2. Call your load_audio function
            # Note: load_audio must return something that can be stored (like a dict or array)
            try:
                audio_array = load_audio(example, min_len=5, max_len=5, sampling_rate=32_000)
                example["audio"] = {"path": example['filepath'], "array": audio_array}
            except Exception as e:
                print(f"Error loading {example['filepath']}: {e}")
                # Return an empty array instead of None to keep Arrow happy
                example["audio"] = {"path": example['filepath'], "array": np.array([], dtype=np.float32)}
            
            return example

        # Apply to all splits
        for split in ["train", "valid", "test"]:
            ds[split] = ds[split].map(process_audio)
        
        

        ds["train"].set_transform(preprocessor)
        ds["valid"].set_transform(test_preprocessor)
        ds["test"].set_transform(test_preprocessor)


        # ads["test"] = extractors.SamExtractor("/home/s.dalal.334/SAM/sam_audio_files/test/sam")["train"]

        model = EfficentNet(num_classes=21)

        args = PyhaTrainingArguments(
            working_dir="working_dir",
            project_name="egci_bioacoustic_shifts"
        )
        args.num_train_epochs = 1
        args.per_device_train_batch_size = 32
        args.per_device_eval_batch_size = 32

        args.eval_steps = math.ceil(len(ds["train"]) / args.per_device_train_batch_size)
        args.learning_rate = 5e-4
        args.dataloader_num_workers = 16
        args.output_dir = f"trained_models/checkpoints/{region}/base"

        args.save_strategy = "steps"
        # save at 15 epochs
        args.save_steps = 0.5

        print(f"args.eval_steps = {args.eval_steps}")

        # matching birdset settings
        # args.weight_decay = 5e-4
        # args.lr_scheduler_type = "cosine"
        # args.warmup_ratio = 5e-2 
        
        args.run_name = str(args.num_train_epochs) +  "/" + parameters["run_name"] + " Base " + parameters["region"]

        trainer = PyhaTrainer(
            model=model,
            dataset=ds,
            training_args=args
        )
        # trainer.evaluate(eval_dataset=hsn_ads["test"], metric_key_prefix="Soundscape_test")
        
        
        print(f"Number reflects number of batches, batch size = {args.per_device_train_batch_size}, so number * batch size reflects number of samples seen during training")

        trainer.train()



        experiment_results[region][parameters["run_name"]] = trainer.evaluate(eval_dataset=ds["test"], metric_key_prefix="Soundscape")

        del model
        del trainer

    # print(region)
    _, _, soundscape_data, _ = load_EGCI(sample=num_samples, region=region, dataset_sub="test_5s")
    _, _, focal_data, focal_samples = load_EGCI(sample=num_samples, region=region, dataset_sub="train")
    process_aug = AugmentAudio(experiment_parameters[0]["augmentation"])
    _, _, aug_focal_data, _ = load_EGCI(
        process_data_func=process_aug ,
        sample=focal_samples, region=region, dataset_sub="train")
    
    print(soundscape_data)
    print(focal_data)
    
    # Format focal and soundscape EGCI for S
    experiment_results[region]["div"] = {}
    experiment_results[region]["div"]["no_aug"] = measure_distrbution_metrics(
        np.vstack((np.array(focal_data[0]), np.array(focal_data[1]))).T,
        np.vstack((np.array(soundscape_data[0]), np.array(soundscape_data[1]))).T,
    )
    experiment_results[region]["div"]["aug"] = measure_distrbution_metrics(
        np.vstack((np.array(aug_focal_data[0]), np.array(aug_focal_data[1]))).T,
        np.vstack((np.array(soundscape_data[0]), np.array(soundscape_data[1]))).T,
    )

    experiment_results[region]["data"]  = {
        "soundscape": soundscape_data,
        "focal": focal_data,
        "augmented": aug_focal_data,
    }
    # Compute EGCI stats with and without data augmentations
    # Save results of both experiments
    with open("e3_results_temp.json", "w") as file:
        json.dump(experiment_results, file, indent=4)