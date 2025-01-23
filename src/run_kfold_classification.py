####
# Apply evaluation suite on a model
#
####

import os
import numpy as np
import torch
import dotenv

from utils.generators import generate_enum_from_string
from utils.globals import TYPES
# Has to be before yucca import to avoid getting warnings
dotenv.load_dotenv()
import argparse
from utils.classification import predict, generate_metrics
from models.bert import BertTokenizer
from models.clip import CLIP, _build_vision, _build_text


class Classification():
    def init_model(self):
        self.checkpoint_clip = os.path.join(self.checkpoint_path, self.checkpoint_name)
        self.model = CLIP.load_from_checkpoint(self.checkpoint_clip, strict=False) # to be argparsed
        self.model.vision = _build_vision(vision_type="swinunetr", use_pretrain=True)
        self.model.text = _build_text(use_bert=True, out_dim=512, use_pretrain=True)
        self.model = self.model.to(self.device)

        self.model.eval()
    def __init__(self, checkpoint_name="tough-sweep/best.ckpt", 
                 test_set="datasets/test_set", 
                 label_dir = os.path.join(os.environ.get("YUCCA_SOURCE"), "GammaKnife", "Brain-TR-GammaKnife-processed"), 
                 checkpoint_path = os.environ.get('YUCCA_MODELS')):
        self.tokenizer = BertTokenizer(os.environ.get("STATES")+"/bert-base-uncased-vocab.txt") # None 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.label_dir = label_dir
        self.test_set = test_set
        self.checkpoint_name = checkpoint_name
        self.checkpoint_path = checkpoint_path
        self.init_model()




if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    dir = "16x_Swin-NTV+NTL"
    models_dir = os.path.join(os.environ.get("YUCCA_MODELS"), dir)
    models = [os.path.join(dir, model) for model in os.listdir(models_dir)]
    types = ['location']

    types =  [generate_enum_from_string(type) for type in types]

    data = "datasets/v2_test_set"
    model_results = {}
    for i in range(len(models)):
        results = {}
        model = models[i]
        for type in types:
            print(f"Running classification task {type} on {model} with data {data}")
            classifier = Classification(checkpoint_name=model, test_set=data)

            # Use these if you want to test the models without pre-trained / clip-trained weights for either vision or text

            # classifier.model.vision = _build_vision(vision_type="swinunetr", use_pretrain=True)
            # classifier.model.text = _build_text(use_bert=True, out_dim=512, use_pretrain=True)
            # classifier.model.vision.to(device)
            # classifier.model.text.to(device)

            predictions = predict(classifier.model, type, classifier.test_set, classifier.device, classifier.tokenizer, verbose=True)
            res = generate_metrics(predictions, type)
            results[type] = res
        
        model_results[model] = results

    for i in range(len(models)):
        model = models[i]
        for type in types:
            print("##############################################################")
            print(f"Classification {type} Results for {model} on {data}")
            for metric in model_results[model][type].keys():
                print(f"        {metric} : {model_results[model][type][metric]}")
            print("--------------------------------------------------------------")    
            print("##############################################################")

        
    # accumulate 
    acc_dict = {}
    for model in model_results.keys():
        for type in model_results[model].keys():
            if type not in acc_dict:
                acc_dict[type] = {}
            for metric in model_results[model][type].keys():
                if metric in acc_dict[type]:
                    acc_dict[type][metric].append(model_results[model][type][metric])
                else: 
                    acc_dict[type][metric] = [model_results[model][type][metric]]


    print("##############################################################")
    for key in acc_dict.keys():
        print(f"Average results over k-fold for classification task {key}")
        for metric,value in acc_dict[key].items():
            print(f"   Average {metric} : {np.array(value).mean()}")
            print(f"   STD {metric} : {np.array(value).std() / np.sqrt(len(value))}")
            print("--------------------------------------------------------------")    
    print("##############################################################")


