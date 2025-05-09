#!/bin/bash

#SBATCH --job-name=tokenization                         # Job name
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

export HF_HUB_OFFLINE=0
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $BASE_DIR

# Download datasets iteratively

datasets=("CC-MAIN-2023-14", "CC-MAIN-2023-23", "CC-MAIN-2023-40", "CC-MAIN-2023-50", "CC-MAIN-2024-10")


python <<'PYCODE'
from huggingface_hub import snapshot_download

for dataset in ["CC-MAIN-2023-14", "CC-MAIN-2023-23", "CC-MAIN-2023-40", "CC-MAIN-2023-50", "CC-MAIN-2024-10"]:
    # Download the dataset
    local_dir = snapshot_download(
        repo_id="HuggingFaceFW/fineweb",
        repo_type="dataset",
        allow_patterns=[f"data/{dataset}/*.parquet"],
        cache_dir=f"/leonardo_scratch/large/userexternal/lcolosi0/minerva/data/raw/fineweb/shrad_{dataset}",
        resume_download=True,
        max_workers=32
    )
    print(f"All Parquet shards for {dataset} are now in: {local_dir}")
PYCODE
