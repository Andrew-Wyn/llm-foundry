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


for i in {1..9}; do 
for j in {1..3}; do

# Set the job name
PARTITION=head
JOB_NAME="rp_"$PARTITION"_""$i"_"$j"

DATA=/leonardo_scratch/large/userexternal/lcolosi0/minerva/data/raw/redpajamas-$PARTITION/$i/$j #$index
OUTPUT_DIR=/leonardo_scratch/large/userexternal/lcolosi0/minerva/data/processed/it/redpajama-$PARTITION/$i/$j/train 


# --- Generate SLURM script rp_n_m_$JOB_NAME.sh ---
cat <<EOF > $MULTIRUN_DIR/$JOB_NAME.sh
#!/bin/bash

#SBATCH --job-name=rp-$PARTITION-$i-$j                 # Job name
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

python scripts/data_prep/convert_dataset_json.py \
    --path "$DATA" \
    --out_root "$OUTPUT_DIR" \
    --split train \
    --concat_tokens 4096 \
    --tokenizer $TOKENIZER \
    --max_tokens 1_000_000_000_000_000 # equivalent to 2000 steps omitt to process whole dataset

EOF
chmod 644 $MULTIRUN_DIR/$JOB_NAME.sh

# sbatch $MULTIRUN_DIR/$JOB_NAME.sh

done
done
