# Revisiting CLIP: Efficient Alignment of 3D MRI and Tabular Data using Domain-Specific Foundation Models

Official Pytorch implementation from the paper

> **Revisiting CLIP: Efficient Alignment of 3D MRI and Tabular Data using Domain-Specific Foundation Models** <br>
> ISBI 2025 <br>
> [Jakob Krogh Petersen*](https://www.linkedin.com/in/jakob-krogh-petersen-656b21129/), [Valdemar Licht*](https://www.linkedin.com/in/valdemar-licht/), [Mads Nielsen](https://scholar.google.de/citations?user=2QCJXEkAAAAJ&hl=en), [Asbjørn Munk](https://asbn.dk)
> 
> Pioneer Centre for AI & University of Copenhagen
>
> \* Equal Contribution

Paper link: [ArXiv]().

![tsne_final](https://github.com/user-attachments/assets/6da89278-a128-42f2-afb6-2353b7c1b153)


## Getting Started

1. Install [Poetry](https://python-poetry.org/docs/).
2. Create environment by calling `poetry install`.
3. Setup environment variables. Run `touch .env`. Then add the following to the new `.env` file:
```
export YUCCA_SOURCE=link/to/datasets
export YUCCA_RAW_DATA=link/to/raw_data
export YUCCA_PREPROCESSED_DATA=link/to/preprocessed_data
export YUCCA_MODELS=link/to/models
export YUCCA_RESULTS=link/to/results
export STATES=src/models/states
```
4. Ensure the states folder exists by running `mkdir -p src/models/states`. This folder contains weights and vocabularies for required models.

5. Create a 'GammaKnife' folder at the chosen YUCCA_SOURCE path. Add the 'Brain-TR-GammaKnife-processed' folder that you download from [here](https://www.cancerimagingarchive.net/collection/brain-tr-gammaknife/) to the 'GammaKnife' folder.

5. Run Task Conversion. Given the setup as above one can run the task conversion task with the corresponding script `bash run_task_conversion.sh`

6. Run Preprocessing. Run the preprocessing step using the corresponding script `bash run_preprocess.sh`

7. Run Training. Run training with the script `bash run_train.sh`. Use arguments `-e` and `-c` to add experiment settings and configuration settings. Use `-f` to train from scratch.
For example, to train from scratch with an experiment locally use:
```
bash run_train.bash -e 16x_swinT_k0 -c local -f
```


## Model checkpoints

We release checkpoints for the best-performing CLIP-trained models for each of the studied vision architectures, as well as the pre-trained models, that we perform the CLIP training.

### CLIP-trained checkpoints

| Vision Model | Parameters (M) |  Checkpoint | 
|--------------|------------|-------------|
| Swin-T       | 8          | [Download](https://zenodo.org/records/14718864/files/16x_Swin-T.zip?download=1) |
| MedNeXt      | 4          | [Download](https://zenodo.org/records/14719239/files/8x_MedNeXt.zip?download=1) |
| ResNet       | 57         | [Download](https://zenodo.org/records/14725305/files/8x_ResNet.zip?download=1) |

### Pre-trained models

| Model | Parameters (M) |  Checkpoint | 
|--------------|------------|-------------|
| Bert         | 110        | [Download](https://huggingface.co/google-bert/bert-base-uncased)<sup>*</sup> |
| Swin-T       | 8          | [Download](https://zenodo.org/records/14719764/files/swinunetr.ckpt?download=1) |
| MedNeXt      | 4          | [Download](https://github.com/asbjrnmunk/amaes)<sup>**</sup> |
| ResNet       | 57         | [Download](https://github.com/asbjrnmunk/amaes)<sup>**</sup> |

*The official model weights can be extracted from HuggingFace. See [here](https://github.com/microsoft/SDNet/blob/master/bert_vocab_files/bert-base-uncased-vocab.txt) for the vocabulary.

**For the MedNeXt and ResNet models, we refer to [AMAES](https://amaes.asbn.dk). 

## Citation

Please use
```
@article{krogh2025clip3d,
  title={Efficient Alignment of 3D MRI and Tabular Data using Domain-Specific Foundation Models},
  author={Petersen, Jakob Krogh and Licht, Johan Valdemar and Nielsen, Mads and Munk, Asbjørn},
  journal={arXiv preprint arXiv:2408.00640},
  year={2025}
}
```
