import pickle as pickle
import os
import torch
from models.clip import CLIP, _build_text, _build_vision
from models.bert import BertTokenizer
from utils.classification import tokenize_text
import torch.nn.functional as F
from utils.globals import ID_TO_LABEL
import copy


embedding_file = os.path.join('src', 'utils', "embedding_dict.pkl")

checkpoint = os.path.join(os.environ.get('YUCCA_MODELS'), "16x_Swin-T/16x_Swin-T_k0_best.ckpt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer(os.environ.get("STATES")+"/bert-base-uncased-vocab.txt")
model = CLIP.load_from_checkpoint(checkpoint)
model.to(device)
text = "not_needed"

dir = os.path.join(os.environ.get('YUCCA_SOURCE'), "v2_test_set")
new_dir = os.path.join(os.environ.get('YUCCA_SOURCE'), "test_set_v2_embedding")



dict_list = []

for filename in os.listdir(dir):
    with open(os.path.join(dir, filename), 'rb') as f:
        case = pickle.load(f)
        case_lesions = case['lesions']
        for lesion in case_lesions:
            lesion_dict = {}
            image = lesion['volume']
            with torch.no_grad():
                image_features, _ = model.get_embeddings(image.to(device), tokenize_text(tokenizer, text).to(device), inference=True)
            image_features =  F.normalize(image_features)

            lesion_dict['case_id'] = lesion['case_id']
            lesion_dict['orig_name'] = lesion['orig_name']
            lesion_dict['main_location'] = lesion['location']
            lesion_dict['locations'] = [ID_TO_LABEL[lesion+1] for lesion in lesion['locations']]
            lesion_dict['embedding'] = image_features.cpu()
            dict_list.append(lesion_dict)

with open(embedding_file, "wb") as f:
    pickle.dump(dict_list, f)


            
