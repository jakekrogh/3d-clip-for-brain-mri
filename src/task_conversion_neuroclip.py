import os
import dotenv
dotenv.load_dotenv()
from sklearn.model_selection import train_test_split
import SimpleITK as sitk
from utils.data_access import get_label_from_filename, get_label_maps, get_mri_file, get_subfolders
from utils.globals import ID_TO_LABEL, LABEL_MAP
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p, subfiles
from yucca.pipeline.task_conversion.utils import generate_dataset_json
from yucca.paths import yucca_raw_data
import pandas as pd

def combine_label_maps(path, save_path):
    # get list of all subfolders
    samples = get_subfolders(path)
    maybe_mkdir_p(save_path)
    for sample in samples:
        shape = sitk.GetArrayFromImage(sitk.ReadImage(join(sample,get_mri_file(sample)))).shape
        accum = np.zeros(shape)
        label_maps = get_label_maps(sample) 
        for label_map in label_maps:
            label_name = get_label_from_filename(label_map)
            label_id = LABEL_MAP[label_name]
            label = sitk.GetArrayFromImage(sitk.ReadImage(join(sample, label_map)))
            accum[label == 1] = label_id
        sample_name = sample.split('/')[-1]
        sample_name = sample_name.replace('_', '-')
        sample_name = sample_name.replace('.', '_')
        sitk.WriteImage(sitk.GetImageFromArray(accum), join(save_path, sample_name+'.nii.gz'))



def convert(path: str, subdir: str = "GammaKnife"):
    """INPUT DATA - Define input path and suffixes"""
    path = join(path, subdir)
    file_suffix = ".nii.gz"

    """ OUTPUT DATA - Define the task name and prefix """
    task_name = "Task002_NEUROCLIPv2"
    task_prefix = "NEUROCLIPv2"

    """ Access the input data. If images are not split into train/test, and you wish to randomly 
    split the data, uncomment and adapt the following lines to fit your local path. """
    images_dir = join(path, "images")
    labels_dir = join(path, "labels")
    reports_dir = join(path, "reports")

    if not os.path.exists(labels_dir):
        combine_label_maps(join(path, "Brain-TR-GammaKnife-processed"), labels_dir )

    samples = subfiles(labels_dir, join=False, suffix=file_suffix)
    train_samples, test_samples = train_test_split(samples, test_size=0.2, random_state=42154)

    images_dir_tr = images_dir_ts = images_dir
    labels_dir_tr = labels_dir_ts = labels_dir



    """ If images are already split into train/test and images/labels uncomment and adapt the following 
    lines to fit your local path."""

    """ Then define target paths """
    target_base = join(yucca_raw_data, task_name)

    target_imagesTr = join(target_base, "imagesTr")
    target_labelsTr = join(target_base, "labelsTr")

    target_imagesTs = join(target_base, "imagesTs")
    target_labelsTs = join(target_base, "labelsTs")


    maybe_mkdir_p(target_imagesTr)
    maybe_mkdir_p(target_labelsTs)
    maybe_mkdir_p(target_imagesTs)
    maybe_mkdir_p(target_labelsTr)

    """Populate Target Directory
    This is also the place to apply any re-orientation, resampling and/or label correction."""

    for sTr in train_samples:
        case_id = sTr[: -len(file_suffix)]
        # Construct the full file path
        
        sitk_img = sitk.ReadImage(join(images_dir_tr, sTr))
        sitk.WriteImage(sitk_img, f"{target_imagesTr}/{task_prefix}_{case_id}_000.nii.gz")
        sitk_label = sitk.ReadImage(join(labels_dir_tr, sTr))
        sitk.WriteImage(sitk_label, f"{target_labelsTr}/{task_prefix}_{case_id}.nii.gz")


    for sTs in test_samples:
        case_id = sTs[: -len(file_suffix)]

        sitk_img = sitk.ReadImage(join(images_dir_ts, sTs))
        sitk.WriteImage(sitk_img, f"{target_imagesTs}/{task_prefix}_{case_id}_000.nii.gz")
        sitk_label = sitk.ReadImage(join(labels_dir_ts, sTs))
        sitk.WriteImage(sitk_label, f"{target_labelsTs}/{task_prefix}_{case_id}.nii.gz")

    generate_dataset_json(
        join(target_base, "dataset.json"),
        target_imagesTr,
        target_imagesTs,
        modalities=("T1",),
        labels=ID_TO_LABEL,
        dataset_name=task_name,
        license="Template",
        dataset_description="Template Dataset",
        dataset_reference="Link to source or similar",
    )


convert(os.environ.get("YUCCA_SOURCE"), "GammaKnife-filtered")