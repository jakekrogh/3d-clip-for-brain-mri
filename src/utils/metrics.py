import json
import os
import pickle
import numpy as np 
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_auc_score, mean_squared_error

def get_mean_rank(gt_ranks : list[int]) -> float:
  return np.mean(np.array(gt_ranks))


def get_auc(y_test, y_scores):
  return roc_auc_score(
     y_true = y_test,
     y_score = y_scores,
  )

def get_auc_all(y_test,y_scores):
  return roc_auc_score(
    y_true=y_test,
    y_score=y_scores,
    average='micro',
    multi_class='ovr'
  )

def get_gts_in_top_k(gts, ranks):
  gts_in_top_k = []
  for gts, ranks in zip(gts, ranks):
    k = len(gts)
    gts_in_top_k.append(list( (set(gts).intersection(set(ranks[:k])), k) ))
  return gts_in_top_k # [ [0], 4]

def get_accuracy_gts_topk(gts_in_top_k):

  return np.array([len(correct)/k if k > 0 else 0 for correct, k in gts_in_top_k]).mean()

def get_auc_individual(y_test,y_scores,labels):
  label_dict = {}
  for i in range(len(labels)):
    label_dict[labels[i]] = i
  
  result_dict = {}
  for (label,label_id) in label_dict.items():
    gts = y_test == label_id
    score = y_scores[:,label_id]
    auc = roc_auc_score(
     y_true = gts,
     y_score = score,
    )
    result_dict[label] = auc
  return result_dict

def get_mse(y_scores):
  return mean_squared_error(
     y_true = np.ones_like(y_scores),
     y_pred = y_scores,
  )


def get_mean_reciprocal_rank(y_true):
  sum = 0
  for elem in y_true:
    sum += 1/(elem)
  return sum/y_true.shape[0]