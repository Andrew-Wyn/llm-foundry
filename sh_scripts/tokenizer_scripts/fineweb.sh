#!/bin/bash

#SBATCH --job-name=2023-40                        # Job name
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
export BASE_DIR=/leonardo/home/userexternal/$(whoami)/minerva/llm-foundry

export PYENV="$BASE_DIR"/llmfoundry-venv
source "$PYENV"/bin/activate

DATA_DIR=/leonardo_scratch/large/userexternal/$(whoami)/minerva/replica
export HF_HOME=/leonardo_scratch/large/userexternal/$(whoami)/.cache/hf_home/

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $BASE_DIR

echo "proicessing CC-MAIN-2023-40"
HF_HUB_OFFLINE=0
python  <<'PYCODE'
from datasets import load_dataset

ds = load_dataset(
  "parquet",
  data_files={"train": "/leonardo_scratch/large/userexternal/lcolosi0/minerva/data/raw/fineweb/shrad_CC-MAIN-2023-40/datasets--HuggingFaceFW--fineweb/snapshots/0f039043b23fe1d4eed300b504aa4b4a68f1c7ba/data/CC-MAIN-2023-40/*.parquet"}
)
ds.save_to_disk("/leonardo_scratch/large/userexternal/lcolosi0/minerva/data/raw/fineweb/hf_dataset/CC-MAIN-2023-40")
PYCODE
echo "done"

# # DATA=/leonardo/prod/data/ai/culturax/2309/it/
# DATA=/leonardo_scratch/large/userexternal/lcolosi0/minerva/data/raw/fineweb/hf_dataset/CC-MAIN-2023-14

# TOKENIZER=sapienzanlp/Minerva-7B-base-v1.0

# OUTPUT_DIR=/leonardo_scratch/large/userexternal/lcolosi0/minerva/data/processed/en/fineweb/CC-MAIN-2023-14
# rm -rf $OUTPUT_DIR # to be on the safe side
# mkdir -p $OUTPUT_DIR # to be on the safe side

# echo "Processing data"
# echo "Saving to $OUTPUT_DIR"

# python $BASE_DIR/scripts/data_prep/convert_dataset_hf.py \
#     --dataset "$DATA" \
#     --out_root "$OUTPUT_DIR" \
#     --concat_tokens 2048 \
#     --split train \
#     --tokenizer "$TOKENIZER" \
#     --no_wrap \
#     --max_tokens 1_000_000_000_000 #equivalent to 2000 steps # omitt to process whole dataset

