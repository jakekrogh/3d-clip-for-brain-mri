import os
import dotenv
# Has to be before yucca import to avoid getting warnings
import pandas as pd
from utils.globals import TYPES

# data access functions
def get_subfolders(path):
    subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    return subfolders
def get_files_with_extension(directory, extension):
    files = [file for file in os.listdir(directory) if file.endswith(extension)]
    return files

def get_mri_file(path):
    files = get_files_with_extension(path, ".nrrd")
    for file in files:
        if "MR_t1" in file:
            return file
        
def get_label_maps(path):
    files = get_files_with_extension(path, ".nrrd")
    res = []
    for file in files:
        if 'MR_t1' in file:
            continue
        elif 'dose' in file:
            continue
        else:
            res.append(file)
    return res

def get_label_from_filename(filename):
    clinical_pd = pd.read_excel(os.path.join("src", "utils", "Brain-TR-GammaKnife_Clinical_Information.xlsx"), sheet_name="lesion_level")
    name = filename[:-5]
    row = clinical_pd[clinical_pd.eq(name).any(axis=1)].head(1)
    try: 
        location = row['location'].values[0]
    except:
        print("EXCEPTION:")
    return location

def compute_gt(type, lesion, data_dict):
    if type == TYPES.GENDER:
        return data_dict['gender']
    if type == TYPES.LOCATION:
        return lesion['location']
    if type == TYPES.LOCATION_DIRECTION:
        loc = lesion['location']
        if loc == 'other':
            return loc
        else:
            return lesion['location'] + ' ' + loc
    if type == TYPES.TUMOR:
        return lesion['lesion_name']

