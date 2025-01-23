import pickle
import lightning as pl
import numpy as np
import torchvision
import logging
import torch
from typing import Literal, Optional
from torch.utils.data import DataLoader, Sampler
from batchgenerators.utilities.file_and_folder_operations import join
from yucca.pipeline.configuration.configure_input_dims import InputDimensionsConfig
from yucca.pipeline.configuration.split_data import SplitConfig
from yucca.pipeline.configuration.configure_plans import PlanConfig
from yucca.data.samplers import InfiniteRandomSampler
from data_loading.YuccaClipDataset import YuccaClipTrainDataset, YuccaClipTestDataset
from utils.globals import TYPES
from utils.globals import LABEL_MAP

class YuccaClipDataModule(pl.LightningDataModule):
    """
    The YuccaDataModule class is a PyTorch Lightning DataModule designed for handling data loading
    and preprocessing in the context of the Yucca project.

    It extends the pl.LightningDataModule class and provides methods for preparing data, setting up
    datasets for training, validation, and prediction, as well as creating data loaders for these stages.

    configurator (YuccaConfigurator): An instance of the YuccaConfigurator class containing configuration parameters.

    composed_train_transforms (torchvision.transforms.Compose, optional): A composition of transforms to be applied to the training dataset. Default is None.
    composed_val_transforms (torchvision.transforms.Compose, optional): A composition of transforms to be applied to the validation dataset. Default is None.

    num_workers (int, optional): Number of workers for data loading. Default is 8.

    pred_data_dir (str, optional): Directory containing data for prediction. Required only during the "predict" stage.

    pre_aug_patch_size (list or tuple, optional): Patch size before data augmentation. Default is None.
        - The purpose of the pre_aug_patch_size is to increase computational efficiency while not losing important information.
        If we have a volume of 512x512x512 and our model only works with patches of 128x128x128 there's no reason to
        apply the transform to the full volume. To avoid this we crop the volume before transforming it.

        But, we do not want to crop it to 128x128x128 before transforming it. Especially not before applying spatial transforms.
        Both because
        (1) the edges will contain a lot of border interpolation artifacts, and
        (2) if we crop to 128x128x128 and then rotate the image 45 degrees or downscale it (zoom out effect)
        we suddenly introduce dark areas where they should not be. We could've simply kept more of the original
        volume BEFORE scaling or rotating, then our 128x128x128 wouldn't be part-black.

        Therefore the pre_aug_patch_size parameter allows users to specify a patch size before augmentation is applied.
        This can potentially avoid dark or low-intensity areas at the borders and it also helps mitigate the risk of
        introducing artifacts during data augmentation, especially in regions where interpolation may have a significant impact.
    """

    def __init__(
        self,
        input_dims_config: InputDimensionsConfig,
        plan_config: PlanConfig,
        splits_config: SplitConfig,
        split_idx: int,
        composed_train_transforms: torchvision.transforms.Compose = None,
        composed_val_transforms: torchvision.transforms.Compose = None,
        num_workers: Optional[int] = None,
        pred_data_dir: str = None,
        pre_aug_patch_size: list | tuple = None,
        train_sampler: Optional[Sampler] = InfiniteRandomSampler,
        val_sampler: Optional[Sampler] = InfiniteRandomSampler,
        n_batches = None,
        epochs = None,
        train_data_dir: str = None,
        tokenizer = None,
        prob_fg = 0.1,
        type = TYPES.LOCATION,
        ignore_empty = False
    ):
        super().__init__()
        self.n_batches = n_batches
        self.epochs = epochs
        # extract parameters
        self.batch_size = input_dims_config.batch_size
        self.patch_size = input_dims_config.patch_size
        self.image_extension = plan_config.image_extension if plan_config else "nii.gz"       
        self.tokenizer = tokenizer
        self.split_idx = split_idx
        self.splits_config = splits_config
        self.train_data_dir = train_data_dir
        self.train_data_dir_text = self.train_data_dir.replace("preprocessed_data", "raw_data").replace("YuccaClipPlanner","reportsTr")
        # Set by initialize()
        self.composed_train_transforms = composed_train_transforms
        self.composed_val_transforms = composed_val_transforms
        self.pre_aug_patch_size = pre_aug_patch_size

        # Set in the predict loop
        self.pred_data_dir = pred_data_dir
        self.pred_data_dir_text = self.pred_data_dir 

        # Set default values

        self.num_workers = max(0, int(torch.get_num_threads() - 1)) if num_workers is None else num_workers
        self.val_num_workers = self.num_workers // 2 if self.num_workers > 0 else self.num_workers
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        self.prob_fg = prob_fg
        self.ignore_empty = ignore_empty
        self.type = type
        logging.info(f"Using {self.num_workers} workers")

    def setup(self, stage: Literal["fit", "test", "predict"]):
        logging.info(f"Setting up data for stage: {stage}")
        expected_stages = ["fit", "test", "predict"]
        assert stage in expected_stages, "unexpected stage. " f"Expected: {expected_stages} and found: {stage}"

        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            assert self.train_data_dir is not None

            self.train_samples = [join(self.train_data_dir, i) for i in self.splits_config.train(self.split_idx)]
            self.val_samples = [join(self.train_data_dir, i) for i in self.splits_config.val(self.split_idx)]
            if len(self.train_samples) < 100:
                logging.info(f"Training on samples: {self.train_samples}")

            if len(self.val_samples) < 100:
                logging.info(f"Validating on samples: {self.val_samples}")

            self.train_dataset = YuccaClipTrainDataset(
                self.train_samples,
                self.train_data_dir_text,
                tokenizer=self.tokenizer,
                composed_transforms=self.composed_train_transforms,
                patch_size=self.pre_aug_patch_size if self.pre_aug_patch_size is not None else self.patch_size,
                type=self.type
            )

            self.val_dataset = YuccaClipTrainDataset(
                self.val_samples,
                self.train_data_dir_text,
                tokenizer=self.tokenizer,
                composed_transforms=self.composed_val_transforms,
                patch_size=self.patch_size,
                type=self.type,
                is_val=True
            )

        if stage == "predict":
            assert self.pred_data_dir is not None, "set a pred_data_dir for inference to work"
            # This dataset contains ONLY the images (and not the labels)
            # It will return a tuple of (case, case_id)
            self.pred_dataset = YuccaClipTestDataset(self.pred_data_dir, suffix=self.image_extension)

    def train_dataloader(self):
        
        logging.info(f"Starting training with data from: {self.train_data_dir}")
        sampler = self.train_sampler(self.train_dataset) if self.train_sampler is not None else None
        return DataLoader(
            self.train_dataset,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler,
            shuffle=False 
        )


    def val_dataloader(self):
        sampler = self.val_sampler(self.val_dataset) if self.val_sampler is not None else None
        return DataLoader(
            self.val_dataset,
            num_workers=self.val_num_workers,
            batch_size=self.batch_size,
            pin_memory=torch.cuda.is_available(),
            sampler=sampler,
        )
        
    def test_dataloader(self):
        return None

    def predict_dataloader(self):
        logging.info("Starting inference")
        return DataLoader(self.pred_dataset, num_workers=self.num_workers, batch_size=1)
    
    def get_sample_labels(self, samples):
        result = [ [] for i in range(len(LABEL_MAP.keys()) - 1) ]
        for i, s in enumerate(samples):
            with open(s[:-3] + "pkl", "rb") as f:
                # Load the object from the pickle file
                data = pickle.load(f)
                labels = np.unique([j['location_type'] for j in data['unique_lesion_maps']]) - 1
                for l in labels:
                    result[l].append(i)
        return result