#!/bin/bash


# Huggingface home
export HF_HOME=/leonardo_scratch/large/userexternal/$(whoami)/.cache/hf_home

# Automatically generate directory and set paths
BASE_DIR=/leonardo/home/userexternal/$(whoami)/minerva/llm-foundry
PYENV="$BASE_DIR"/llmfoundry-venv
LOG_DIR="$BASE_DIR"/logs/data-processing
mkdir -p $LOG_DIR

MULTIRUN_DIR=$BASE_DIR/sh_scripts/tokenizer_scripts/multirun/
TOKENIZER=sapienzanlp/Minerva-7B-base-v1.0

datasets_names=("CC-MAIN-2023-14" "CC-MAIN-2023-23" "CC-MAIN-2023-40" "CC-MAIN-2023-50" "CC-MAIN-2024-10")

for i in "${datasets_names[@]}"; do 

# Set the job name
PARTITION=fineweb
JOB_NAME=$PARTITION"_""$i"

DATA=/leonardo_scratch/large/userexternal/lcolosi0/minerva/data/raw/fineweb/hf_dataset/$i #$index
OUTPUT_DIR=/leonardo_scratch/large/userexternal/lcolosi0/minerva/data/processed/en/$PARTITION-$i/train 


# --- Generate SLURM script rp_n_m_$JOB_NAME.sh ---
cat <<EOF > $MULTIRUN_DIR/$JOB_NAME.sh
#!/bin/bash

#SBATCH --job-name=$PARTITION-$i                 # Job name
#SBATCH --output=/leonardo/home/userexternal/lcolosi0/minerva/llm-foundry/logs/data-processing/%x-%j.out
#SBATCH --error=/leonardo/home/userexternal/lcolosi0/minerva/llm-foundry/logs/data-processing/%x-%j.err

#SBATCH -A mnrv_bblscp                  # account to charge
#SBATCH -p boost_usr_prod               # partition to use

#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --cpus-per-task=32              # number of threads per task
#SBATCH --time 24:00:00                 # format: HH:MM:SS
#SBATCH --gres=gpu:0                    # number of gpus per node


# Set up directories paths
source "$PYENV"/bin/activate

DATA_DIR=/leonardo_scratch/large/userexternal/$(whoami)/minerva/replica
export HF_HOME=/leonardo_scratch/large/userexternal/$(whoami)/.cache/hf_home/

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $BASE_DIR

mkdir -p $OUTPUT_DIR
echo "Saving to $OUTPUT_DIR"

python $BASE_DIR/scripts/data_prep/convert_dataset_hf.py \
    --dataset "$DATA" \
    --out_root "$OUTPUT_DIR" \
    --concat_tokens 2048 \
    --split train \
    --tokenizer "$TOKENIZER" \
    --no_wrap \
    --max_tokens 1_000_000_000_000 

EOF
chmod 644 $MULTIRUN_DIR/$JOB_NAME.sh

# sbatch $MULTIRUN_DIR/$JOB_NAME.sh

done
