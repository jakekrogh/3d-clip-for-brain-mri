import pickle as pickle
import os
import torch
from models.clip import CLIP
from models.bert import BertTokenizer
from utils.classification import tokenize_text
from utils.generators import generate_location
import torch.nn.functional as F
from utils.globals import ID_TO_LABEL
import copy
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import csv
import random

savefigs = False

def plot_sorted_distances(distances, labels, prompt, name, n_samples):
    labeled_distances = list(zip(distances, labels))
    labeled_distances.sort(key=lambda x: x[0])

    sorted_distances, sorted_labels = zip(*labeled_distances)
    
    # accuracy
    accuracy = sum([label == name for label in sorted_labels[:n_samples]])/n_samples

    # MAP
    relevant = [1 if label == name else 0 for label in sorted_labels[:n_samples]]
    precision_at_k = [sum(relevant[:k+1])/(k+1) for k in range(n_samples)]
    map_score = sum(precision_at_k) / n_samples

    # MRR 
    reciprocal_ranks = [1/(rank+1) for rank, label in enumerate(sorted_labels[:n_samples]) if label == name]
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0
    
    print(f'Location: {name}, Number of samples: {n_samples}, Accuracy: {accuracy:.2f}% MAP: {map_score:.2f}, MRR: {mrr:.2f}\n')
    # plot
    if savefigs:
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(sorted_distances)), sorted_distances, tick_label=sorted_labels)
        
        plt.xlabel('Labels')
        plt.ylabel('Distances')
        plt.title('Prompt: ' + prompt)
        plt.xticks(rotation=45, ha='right') 
        plt.tight_layout()
        
        legend_text = f'Num Samples: {n_samples}\nAccuracy: {accuracy:.2f}%\nMAP: {map_score:.2f}\nMRR: {mrr:.2f}'
        plt.legend([legend_text], loc='upper left', frameon=True)
        
        plt.savefig('dist_graphs/'+name+'.png')
    return mrr, map_score, accuracy

embedding_file = os.path.join('src', 'utils', "embedding_dict.pkl")

with open(embedding_file, "rb") as f:
    vision_embeddings = pickle.load(f)

vision_embeddings_array = np.squeeze(np.array([element['embedding'] for element in vision_embeddings]), axis=1)
vision_label_array =np.array([element['main_location'] for element in vision_embeddings])



locations = ['frontal', 'cerebellar', 'occipital', 'parietal', 'temporal']

# prompts 
prompts = [generate_location(location) for location in locations]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_sample = torch.rand(size=(1,1,64,64,64))

n_samples = [sum([sample['main_location'] == location for sample in vision_embeddings]) for location in locations]

models_dir = os.listdir(os.environ.get('YUCCA_MODELS'))
done_models = [name[:-4] for name in os.listdir("information_retrieval")]


for model in models_dir:
    if model in done_models:
        continue
    results = []
    model_scores = {"mmr": [], 
                    "map": [], 
                    "accuracy": []}
    splits = os.listdir(os.path.join(os.environ.get('YUCCA_MODELS'), model))
    for split in splits:
        checkpoint = os.path.join(os.environ.get('YUCCA_MODELS'), model, split)
        tokenizer = BertTokenizer(os.environ.get("STATES") + "/bert-base-uncased-vocab.txt")
        model_instance = CLIP.load_from_checkpoint(checkpoint, strict=False)
        model_instance.to(device)
        for i in range(5):
            with torch.no_grad():
                _, prompt_embedding = model_instance.get_embeddings(vision_sample.to(device), tokenize_text(tokenizer, prompts[i]).to(device))

                # compute distances
                dist_array = np.linalg.norm(vision_embeddings_array - prompt_embedding.cpu().numpy(), axis=1)


                # plot and compute scores
                mmr, map_score, acc = plot_sorted_distances(dist_array, vision_label_array, prompts[i], locations[i], n_samples[i])
                model_scores["mmr"].append(mmr)
                model_scores["map"].append(map_score)
                model_scores["accuracy"].append(acc)
                results.append({
                    "model": model,
                    "split": split,
                    "location": locations[i],
                    "mmr": mmr,
                    "map": map_score,
                    "accuracy": acc
                })
        
        # append results for the current split
        results.append({
            "model": model,
            "split": split,
            "location": "average",
            "mmr": np.mean(model_scores["mmr"]),
            "map": np.mean(model_scores["map"]),
            "accuracy": np.mean(model_scores["accuracy"]),
            "sem" : np.array(model_scores['accuracy']).std() / np.sqrt(len(model_scores['accuracy']))
        })
        results.append({})

    results_df = pd.DataFrame(results)
    results_df.to_csv(f'information_retrieval/{model}.csv', index=False)





            
