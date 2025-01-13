# NeuroCLIP

## Usage

### 0. Setup Python environment with Poetry

### 1. Setup environment variables.
Setup a .env file at the root directory that contians variables pointing to paths to relevant directories. A simple example
of a potential .env file can be seen here:

export YUCCA_SOURCE=datasets
export YUCCA_RAW_DATA=raw_data
export YUCCA_PREPROCESSED_DATA=preprocessed_data
export YUCCA_MODELS=models
export YUCCA_RESULTS=results
export STATES=src/models/states

### 2. Setup model states
In 'src/models' add the folder 'states' which contains weights and vocabularies for required models. 

### 3. Add dataset to directory.
The setup assumes the name of the dataset folder to be 'GammaKnife-filtered', and located in the path pointed to by the 
YUCCA_SOURCE env variable. It should contain an 'images' and 'labels' folder with corresponding images and label maps in the nii.gz format. This is our personal restructuring of the original dataset to fit yucca standards. The project also needs the original dataset folder called  'Brain-TR-GammaKnife-processed'.  


### 4. Run Task Conversion
Given the setup as above one can run the task conversion task with the corresponding script 'run_task_conversion.sh'

### 5. Run Preprocessing
Run the preprocessing step using the corresponding script 'run_preprocess.sh'

### 6. Run Training
Run training with the script 'run_train.sh'. Use arguments -e and -c to add experiment settings and configuration settings. Use -f to train from scratch.
Ex train from scratch on a debug experiment locally:
'sh run_train.bash -e debug_val -c local -f'

### External files needed
- Dataset folder 'GammaKnife-filtered'
- States folder 'states'
