import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from utils.metrics import get_accuracy_gts_topk, get_auc, get_auc_all, get_auc_individual, get_gts_in_top_k, get_mean_rank, get_mean_reciprocal_rank, get_mse
from utils.data_access import compute_gt
from utils.generators import generate_texts
from utils.globals import TYPES, types
from models.simple_tokenizer import tokenize
from einops import rearrange, reduce
from torch import nn, einsum, log
def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom
def max_neg_value(dtype):
    return -torch.finfo(dtype).max
def tokenize_text(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
def predict(model, type, test_set_path, device, tokenizer = None, verbose = False):
  gt_ranks = []
  y_scores = []
  y_scores_all = []
  gts = []
  labels = types[type]['labels']
  locations = []
  rankings = []
  for case in os.listdir(test_set_path):
      data_dict = pickle.load(open(os.path.join(test_set_path, case), "rb"))
      case_lesions = data_dict['lesions']
      texts = generate_texts(data_dict, type)
      if tokenizer:  
        texts = torch.stack([tokenize_text(tokenizer, text) for text in texts], dim=0).squeeze(dim=1)
      else:
        texts = torch.stack([tokenize(text) for text in texts], dim=0).squeeze(dim=1)
      for lesion in case_lesions:
          image = lesion['volume'] 
          with torch.no_grad():
              image_features, text_features = model.get_embeddings(image.to(device), texts.to(device), inference=True)
          image_features =  F.normalize(image_features)
          text_features =  F.normalize(text_features)
          similarity = (model.logit_scale * (image_features @ text_features.T) ).softmax(dim=-1)
          values, indices = similarity[0].topk(len(labels)) # rank probs
          gt = compute_gt(type, lesion, data_dict)
          gt_index = labels.index(gt) # get probs for gt
          gts.append(gt_index)
          # get rank
          rank = list(indices).index(gt_index) + 1
          gt_ranks.append(rank)
          # save probability scores
          y_scores.append(values[gt_index].detach().cpu().numpy())
          y_scores_all.append(similarity[0].detach().cpu().numpy()) 
          rankings.append(indices.cpu().numpy()) 
          if type == TYPES.LOCATION:
              locs_in_patch = np.array(lesion['locations']).astype(int)
              locations.append(locs_in_patch)
          if verbose:
            print(f"\nPredicted {case} gt as the {rank} most likely.\n")
            for value, index in zip(values[:rank], indices[:rank]):
                print(f"probability:{value}, {type}:{labels[index]}")
  results = {}
  results['type'] = type
  results['gt_ranks'] = gt_ranks
  results['gts'] = gts
  results['y_scores'] = y_scores
  results['y_scores_all'] = y_scores_all
  results['locations'] = locations
  results['rankings'] =rankings
  return results
def generate_metrics(prediction_dict, type = TYPES.LOCATION):
    result_dict = {}
    ranks = prediction_dict['gt_ranks']
    gts = prediction_dict['gts']
    y_scores = prediction_dict['y_scores']
    y_scores_all = prediction_dict['y_scores_all']
    locations = prediction_dict['locations']
    rankings = prediction_dict['rankings']
    labels = types[type]['labels']
    mean_rank = get_mean_rank(ranks)
    mse = get_mse(np.asarray(y_scores))
    mrr = get_mean_reciprocal_rank(np.array(ranks))
    result_dict['mean_rank'] = mean_rank
    result_dict['mse'] = mse
    result_dict['mrr'] = mrr
    if prediction_dict['type'] == TYPES.LOCATION:
      auc_individual = get_auc_individual(np.array(gts), np.array(y_scores_all),labels)
      auc = get_auc_all(np.array(gts), np.array(y_scores_all))
      precision_3 = len([x for x in ranks if x <= 3])/len(ranks)
      precision_1 = len([x for x in ranks if x <= 1])/len(ranks)
      result_dict['Precision @ 3'] = precision_3
      result_dict['Precision @ 1'] = precision_1 
      gts_in_topk = get_gts_in_top_k(locations, rankings)
      result_dict['ACC_GTS_IN_TOP_K'] = get_accuracy_gts_topk(gts_in_topk)
      result_dict['AUC AVERAGE'] = auc
      for (label,label_auc) in auc_individual.items():
          result_dict['AUC ' + label] = label_auc
    return result_dict