#!/bin/bash
RED='\033[1;31m'
YELLOW='\033[1;33m'
IWhite='\033[0;97m'
IGreen='\033[0;92m'
NC='\033[0m' # No Color

# stop on error
set -e

# Usage function
usage() {
   echo -e "Usage: $0 -e <experiment_name> -c <config_name> [-f]\n
    Options:
       -e     name of experiment file in '/experiments' folder
       -c     name of configuration file in '/configs' folder
       -f     don't continue from the most recent checkpoint, but start a new version from scratch"
    exit 1
}

# Initialize from_scratch variable
from_scratch=0

# Parse command-line arguments
while getopts ":e:c:f" opt; do
    case $opt in
        e) experiment_name="$OPTARG" ;;
        c) config_name="$OPTARG" ;;
        f) from_scratch=1 ;;
        \?) usage ;;
    esac
done

# Check if both arguments are provided
if [ -z "$experiment_name" ] || [ -z "$config_name" ]; then
    usage
fi

if [ $from_scratch == 1 ]; then
    printf "${RED}Warning:${IWhite} Training from scratch, even if the script could continue!${NC}\n"
    continue="--new_version"
else
    printf "${YELLOW}Continuing training of most recent version if possible${NC}\n"
    continue=""
fi

echo -e "Submitting with experiment ${IWhite}'$experiment_name'${NC} and config ${IWhite}'$config_name'${NC}"

# Construct paths
experiment_file="experiments/$experiment_name"
config_file="configs/$config_name"

# Check if the files exist
if [ ! -f "$experiment_file" ] || [ ! -f "$config_file" ]; then
    echo "Experiment or Config file does not exist."
    exit 1
fi

# Check if the files end with newlines
if [ ! "$(tail -c1 "$experiment_file" | wc -l)" -eq 1 ]; then
    echo "The experiment file does not end with a newline."
    exit 1
fi

if [ ! "$(tail -c1 "$config_file" | wc -l)" -eq 1 ]; then
    echo "The config file does not end with a newline."
    exit 1
fi

# Initialize the argument string for src/main.py
args=""

# Read each line in the experiment file
while IFS='=' read -r key value; do
    # Skip lines that start with '#'
    [[ $key == \#* ]] || [[ -z $key ]] && continue

    # Add all other args
    if [[ $value == "" ]]; then
        args+=" --$key"
    else
        args+=" --$key=$value"
    fi
done < "$experiment_file"

# Save job script
current_datetime=$(date "+%Y%m%d_%H%M%S")
mkdir -p jobs # create jobs folder if doesnt exist
job_script="jobs/${experiment_name}_${config_name}_${current_datetime}.job"
touch $job_script
cat "$config_file" > "$job_script"

configuration_args="\$COMPILE_ARGS $continue --num_workers="\$NUM_WORKERS" --num_devices="\$NUM_DEVICES" --experiment=$experiment_name"

# Write the command to run the experiment
cat << EOF >> "$job_script"
\$INSTALL_CMD
\$RUN_ENV src/train.py $args $configuration_args
EOF

# Check contents of the config file to determine the scheduler
if [[ $config_file == *"local"* ]]; then
    bash "$job_script"
elif [[ $config_file == *"hendrix"* ]]; then
    sbatch "$job_script"
    echo -e "${IGreen}Success!${NC} Experiment submitted with ${IWhite}sbatch${NC}."
elif [[ $config_file == *"gbar"* ]]; then
    bsub < "$job_script"
    echo -e "${IGreen}Success!${NC} Experiment submitted with ${IWhite}bsub${NC}"
else
    echo "Unknown scheduler configuration."
    exit 1
fi