####
# Train a Linear probe on k folds, and apply evaluation suite to it.
#
####

import os
import torch
import dotenv
from models.mednext import MedNeXt
from models.unet import UNet
from models.resunet import resunet
# Has to be before yucca import to avoid getting warnings
dotenv.load_dotenv()
import numpy as np
from models.clip import SwinUNETR_SSL, CLIP
from sklearn.linear_model import LogisticRegression
import pickle
from utils.metrics import get_auc, get_auc_individual, get_mse, get_mean_rank, get_auc_all
from utils.globals import gender_to_id, LABEL_MAP, types, TYPES
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(model_path):
    return CLIP.load_from_checkpoint(os.environ.get("YUCCA_MODELS")+'/' + model_path, strict=False)

test_set_path = ["datasets/v2_test_set"]*5
train_set_path = ["datasets/v2_train_set_k0"]*5


dir = "16x_Swin-NTV+NTL"
models_dir = os.path.join(os.environ.get("YUCCA_MODELS"), dir)
models = [os.path.join(dir, model) for model in os.listdir(models_dir)]

res = {}
accuracies = []
aucs = []
#### BEGIN PROBING ####
for train_set, test_set, model in zip(train_set_path, test_set_path, models):
    print(f"RUNNING PROBE ON TRAIN SET {train_set} AND TEST SET {test_set}")
    model = init_model(model).vision
    model.eval()
    train_image_features, train_labels = [], []
    test_image_features, test_labels = [], []

    for case in os.listdir(train_set):
      case_pkl = pickle.load(open(os.path.join(train_set, case), "rb"))
      case_lesions = case_pkl['lesions']
      for lesion in case_lesions:
        image, labels = lesion['volume'], LABEL_MAP[lesion['location']] - 1, # -1 to remove background id from label space 
        with torch.no_grad():
          image_features = torch.flatten(model(torch.Tensor(image).to(device)), start_dim=1).to(device)
        train_image_features.append(image_features.cpu().numpy().flatten())
        train_labels.append(labels)


    for case in os.listdir(test_set):
      case_pkl = pickle.load(open(os.path.join(test_set, case), "rb"))
      case_lesions = case_pkl['lesions']
      for lesion in case_lesions:
        image, labels = lesion['volume'], LABEL_MAP[lesion['location']] - 1, # -1 to remove background id from label space 
        with torch.no_grad():
          image_features = torch.flatten(model(torch.Tensor(image).to(device)), start_dim=1).to(device)
        test_image_features.append(image_features.cpu().numpy().flatten())
        test_labels.append(labels)

    train_image_features = np.array(train_image_features)
    train_labels = np.array(train_labels)
    test_image_features = np.array(test_image_features)
    test_labels = np.array(test_labels)

    classifier = LogisticRegression(random_state=0, max_iter=1000, verbose=1)
    classifier.fit(train_image_features, train_labels)
    predictions = classifier.predict(test_image_features)
    logits = classifier.predict_proba(test_image_features)


    accuracy = np.mean(predictions == test_labels).astype(float)
    auc_individual = get_auc_individual(test_labels, logits, types[TYPES.LOCATION]['labels'])
    for (label,label_auc) in auc_individual.items():
          if label in res:
            res[label].append(label_auc)
          else:
            res[label] = [label_auc]
    auc = get_auc_all(test_labels, logits) 
    aucs.append(auc)
    accuracies.append(accuracy)

    print(f"Accuracy: {accuracy}%")
    print("----------------------------------")
    print(f"AUC: {auc }")
    print("----------------------------------")
    print(f"Mean Rank: {get_mean_rank(predictions)}")
    print("----------------------------------")
    for key,value in auc_individual.items():
      print(f"AUC {key}: {value}")
      print("----------------------------------")





average_auc = np.array(aucs).mean()
std_auc = np.array(aucs).std() / np.sqrt(len(aucs))
average_accs = np.array(accuracies).mean()
std_accs = np.array(accuracies).std() / np.sqrt(len(accuracies))


print("###################################")
print(f"Mean accuracy: {average_accs}%")
print(f"STD accuracy: {std_accs}%")
print("----------------------------------")
print(f"Mean AUC: {average_auc }")
print(f"STD  AUC +/-: {std_auc }")
print("----------------------------------")
for key,value in res.items():
  print(f"AVERAGE AUC {key}: {np.array(value).mean()}")
  print(f"SEM AUC {key} +/-: {np.array(value).std() / np.sqrt(len(value))}")
  print("----------------------------------")

