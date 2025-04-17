#!/bin/bash
#SBATCH --job-name=cpt-data-processing                          # Job name
#SBATCH --output=/leonardo/home/userexternal/lmoroni0/__Work/llm-foundry/logs/data-processing/wikipedia_it-%x-%j.out         # Name of stdout output file
#SBATCH --error=/leonardo/home/userexternal/lmoroni0/__Work/llm-foundry/logs/data-processing/wikipedia_it-%x-%j.err          # Name of stderr error file

#SBATCH --nodes=1                       # number of nodes
#SBATCH --ntasks-per-node=1             # number of tasks per node
#SBATCH --cpus-per-task=32              # number of threads per task
#SBATCH --time 24:00:00                 # format: HH:MM:SS
#SBATCH --gres=gpu:0                    # number of gpus per node
##SBATCH --qos=qos_llm_min               # quality of service

#SBATCH -A IscrB_medit                  # account to charge
#SBATCH -p boost_usr_prod               # partition to use

##SBATCH --mail-type=ALL                             # send email on job start, end and fail
##SBATCH --mail-user=conia@diag.uniroma1.it          # email address to send to

# Load necessary modules
module load python

# Set up HuggingFace
export HF_HOME="${CINECA_SCRATCH}/hf_cache"
export HF_DATASETS_CACHE="${CINECA_SCRATCH}/hf_cache"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd  /leonardo/home/userexternal/lmoroni0/__Work/llm-foundry

# load virtual environment
source .env/bin/activate

python scripts/data_prep/convert_dataset_json.py \
  --path $SCRATCH/data/json/cpt/it/wikipedia \
  --out_root $SCRATCH/data/mds/cpt/it/wikipedia/train \
  --split train \
  --concat_tokens 4096 --tokenizer $SCRATCH/models/minerva-7B-base \
  --max_tokens 100_000_000_000