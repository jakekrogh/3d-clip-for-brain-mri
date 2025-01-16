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
3. Setup environment variables.
Setup a .env file at the root directory that contians variables pointing to paths to relevant directories. A simple example
of a potential .env file can be seen here:
```
export YUCCA_SOURCE=link/to/datasets
export YUCCA_RAW_DATA=link/to/raw_data
export YUCCA_PREPROCESSED_DATA=link/to/preprocessed_data
export YUCCA_MODELS=link/to/models
export YUCCA_RESULTS=link/to/results
export STATES=src/models/states
```
4. In 'src/models' add the folder 'states' which contains weights and vocabularies for required models.
```
mkdir -p src/models/states
```   
5. Add dataset to directory.
Download the data from [here](https://www.cancerimagingarchive.net/collection/brain-tr-gammaknife/).The setup assumes the name of the dataset folder to be 'GammaKnife-filtered', and located in the path pointed to by the `YUCCA_SOURCE` env variable. It should contain an 'images' and 'labels' folder with corresponding images and label maps in the nii.gz format. This is our personal restructuring of the original dataset to fit yucca standards. The project also needs the original dataset folder called  'Brain-TR-GammaKnife-processed'.  

5. Run Task Conversion
Given the setup as above one can run the task conversion task with the corresponding script `bash run_task_conversion.sh`

6. Run Preprocessing
Run the preprocessing step using the corresponding script `bash run_preprocess.sh`

7. Run Training
Run training with the script `bash run_train.sh`. Use arguments `-e` and `-c` to add experiment settings and configuration settings. Use `-f` to train from scratch.
For example, to train from scratch on a debug experiment locally use:
```
bash run_train.bash -e debug_val -c local -f
```
