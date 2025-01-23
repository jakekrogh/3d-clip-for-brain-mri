import os
import dotenv
dotenv.load_dotenv()
import SimpleITK as sitk
import pandas as pd
from utils.data_access import get_label_from_filename, get_label_maps, get_mri_file, get_subfolders
from utils.globals import LABEL_MAP, course_pd, lesion_pd
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
import pandas as pd

def convert_to_niigz(input_path, output_path):
    # Read the image
    image = sitk.ReadImage(input_path)

    # Write the image
    sitk.WriteImage(image, output_path)


def combine_label_maps(sample_folder, save_path, sample_name):
    # get list of all subfolders
    maybe_mkdir_p(save_path)
    shape = sitk.GetArrayFromImage(sitk.ReadImage(join(sample_folder,get_mri_file(sample_folder)))).shape
    accum = np.zeros(shape)
    label_maps = get_label_maps(sample_folder) 
    for label_map in label_maps:
        label_name = get_label_from_filename(label_map)
        label_id = LABEL_MAP[label_name]
        label = sitk.GetArrayFromImage(sitk.ReadImage(join(sample_folder, label_map)))
        accum[label == 1] = label_id
    sitk.WriteImage(sitk.GetImageFromArray(accum), join(save_path, sample_name))


if __name__ == "__main__":
    dataset_path = os.path.join(os.environ.get("YUCCA_SOURCE"), "GammaKnife")
    mri_path = os.path.join(dataset_path, "Brain-TR-GammaKnife-processed")
    images_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    uncategorized_files = lesion_pd[lesion_pd['location'] == "uncategorized"]['Lesion Name in NRRD files'].tolist()
    for root, dirs, files in os.walk(mri_path):
        for dir in dirs:
            patient_id = int(dir[3:6])
            course_level = int(dir[7])
            is_filtered = course_pd.loc[(course_pd['Course #'] == course_level) & (course_pd['unique_pt_id'] == patient_id)].empty
            if is_filtered:
                continue
            for file in os.listdir(os.path.join(mri_path, dir)):
                if file.split('/')[-1][:-5] in uncategorized_files:
                    file_path = os.path.join(mri_path,dir, file)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                if "MR_t1" in file:
                        input_file = os.path.join(root,dir, file)
                        cl = file[3:8].replace("_", "-")
                        file_name = f"GK_{cl}.nii.gz"
                        convert_to_niigz(input_file, os.path.join(images_dir, file_name))
            combine_label_maps(os.path.join(mri_path,dir), labels_dir, f"GK_{patient_id}-{course_level}.nii.gz")



