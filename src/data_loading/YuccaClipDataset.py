import numpy as np
import torch
from typing import Union, Optional
import logging
from batchgenerators.utilities.file_and_folder_operations import subfiles, load_pickle, isfile
from image_processing.transforms.cropping_and_padding import CropPad
from yucca.data.augmentation.transforms.formatting import NumpyToTorch
import re
from models.bert import BertTokenizer
from utils.globals import TYPES
from utils.generators import generate_age, generate_gender, generate_random_age, generate_random_gender, generate_random_locations, generate_description
from utils.wandb import get_gif_local
import matplotlib.pyplot as plt
import random

class YuccaClipTrainDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        samples: list,
        text_samples_dir: str,
        patch_size: list | tuple,
        keep_in_ram: Union[bool, None] = None,
        label_dtype: Optional[Union[int, float]] = None,
        composed_transforms=None,
        tokenizer = None,
        is_val = False,
        type = None
    ):
        self.text_samples_dir = text_samples_dir
        self.all_cases = samples
        self.composed_transforms = composed_transforms
        self.patch_size = patch_size
        self.label_dtype = label_dtype
        self.tokenizer = tokenizer
        self.already_loaded_cases = {}
        self.is_val = is_val
        self.type = type
        self.debug = False

        # for segmentation and classification we override the default None
        # because arrays are saved as floats and we want them to be ints.
        if self.label_dtype is None:
            self.label_dtype = torch.int32

        self.croppad = CropPad(patch_size=self.patch_size, p_oversample_foreground=1.0)
        self.to_torch = NumpyToTorch(label_dtype=self.label_dtype)

        self._keep_in_ram = keep_in_ram

    @property
    def keep_in_ram(self):
        if self._keep_in_ram is not None:
            return self._keep_in_ram
        if len(self.all_cases) < 50:
            self._keep_in_ram = True
        else:
            logging.debug("Large dataset detected. Will not keep cases in RAM during training.")
            self._keep_in_ram = False
        return self._keep_in_ram

    def load_and_maybe_keep_pickle(self, path):
        path = path + ".pkl"
        if not self.keep_in_ram:
            return load_pickle(path)
        if path in self.already_loaded_cases:
            return self.already_loaded_cases[path]
        self.already_loaded_cases[path] = load_pickle(path)
        return self.already_loaded_cases[path]

    def load_and_maybe_keep_volume(self, path):
        path = path + ".npy"
        if not self.keep_in_ram:
            if isfile(path):
                try:
                    return np.load(path, "r")
                except ValueError:
                    return np.load(path, allow_pickle=True)
            else:
                print("uncompressed data was not found.")

        if isfile(path):
            if path in self.already_loaded_cases:
                return self.already_loaded_cases[path]
            try:
                self.already_loaded_cases[path] = np.load(path, "r")
            except ValueError:
                self.already_loaded_cases[path] = np.load(path, allow_pickle=True)
        else:
            print("uncompressed data was not found.")

        return self.already_loaded_cases[path]

    def __len__(self):
        return len(self.all_cases)

    def __getitem__(self, idx):
        case = self.all_cases[idx]
        data = self.load_and_maybe_keep_volume(case)
        data_dict = {"file_path": case}  # metadata that can be very useful for debugging.
        label = np.array([data[-1:][0]]).astype(int)        
        data_dict.update({"image": np.array([data[:-1][0]]), "label": label})
        data_dict, metadata = self._transform(data_dict, case)
        uniques = data_dict['uniques']
        del data_dict['uniques']
        label = data_dict['label']
        assert label.shape == data_dict['image'].shape

        if not self.type:
            text = generate_description(metadata, uniques)
        elif self.type.value == TYPES.LOCATION.value:
            text = generate_random_locations(uniques)
        elif self.type.value == TYPES.AGE.value:
            text = generate_random_age(metadata)
        elif self.type.value == TYPES.GENDER.value:
            text = generate_random_gender(metadata)
        else:
            text = generate_description(metadata, uniques)
        data_dict["raw_text"] = text # save the raw text for gif generation

        if self.tokenizer == None:
            data_dict.update({"text": tokenize(text) })
        else:
            tokenized_text = self.tokenizer.tokenize(text)
            input_ids = np.array((self.tokenizer.convert_tokens_to_ids(tokenized_text)))
            data_dict.update({"text": input_ids})
        return self.to_torch(data_dict)


    def _transform(self, data_dict, case):
        seed = random.random()
        if self.debug:
            pre_crop_image = np.copy(data_dict["image"])
            save_fig(pre_crop_image[0,:,:,100], "base", "pre_crop", seed)
        metadata = self.load_and_maybe_keep_pickle(case)
        data_dict = self.croppad(data_dict, metadata)
        if self.debug:
            crop_image = np.copy(data_dict["image"])
            save_fig(crop_image[0,:,:,48], "patch", "post_crop", seed, size=(0.64,0.64))
        if self.composed_transforms is not None:
            data_dict = self.composed_transforms(data_dict)
        if self.debug:
            aug_image = np.copy(data_dict["image"])
            save_fig(aug_image[0,:,:,48], "aug", "post_aug", seed, size=(0.64,0.64))
        self.debug = False
        return data_dict, metadata
        ### 

def save_fig(image, folder, name, seed, size=(1.92, 1.92)):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.gcf().set_size_inches(size)
    plt.tight_layout()
    plt.savefig(f"{folder}/example_{name}{seed}.png", dpi=200, bbox_inches='tight', pad_inches=0)

class YuccaClipTestDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data_dir, suffix="nii.gz"):
        self.data_path = raw_data_dir
        self.suffix = suffix
        self.unique_cases =  subfiles(self.data_path, suffix=self.suffix, join=False)
        assert len(self.unique_cases) > 0, f"No cases found in {self.data_path}. Looking for files with suffix: {self.suffix}"

    def __len__(self):
        return len(self.unique_cases)

    def __getitem__(self, idx):
        # Here we generate the paths to the cases along with their ID which they will be saved as.
        # we pass "case" as a list of strings and case_id as a string to the dataloader which
        # will convert them to a list of tuples of strings and a tuple of a string.
        # i.e. ['path1', 'path2'] -> [('path1',), ('path2',)]
        case_id = self.unique_cases[idx]
        return [self.data_path + "/" + case_id], case_id


