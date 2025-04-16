#!/bin/bash

#SBATCH --job-name=train-tokenizer                        # Job name
#SBATCH --output=/leonardo/home/userexternal/lcolosi0/minerva-pretraining/logs/data-processing/%x-%j.out         # Name of stdout output file
#SBATCH --error=/leonardo/home/userexternal/lcolosi0/minerva-pretraining/logs/data-processing/%x-%j.err          # Name of stderr error file
#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --cpus-per-task=32              # number of threads per task
#SBATCH --time 24:00:00                 # format: HH:MM:SS
#SBATCH --gres=gpu:0                    # number of gpus per node
##SBATCH --qos=qos_llm_min               # quality of service

#SBATCH -A mnrv_bblscp                  # account to charge
#SBATCH -p boost_usr_prod               # partition to use

# Set up directories paths
export BASE_DIR=/leonardo/home/userexternal/$(whoami)/minerva/llm-foundry

export PYENV="$BASE_DIR"/llmfoundry-venv
source "$PYENV"/bin/activate

DATA_DIR=/leonardo_scratch/large/userexternal/$(whoami)/minerva/replica
export HF_HOME="$DATA_DIR"/.cache/hf_home/

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $BASE_DIR

python "$BASE_DIR"/tests/test_minerva.py
