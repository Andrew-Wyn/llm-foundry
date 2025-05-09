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

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $BASE_DIR

DATA=/leonardo/prod/data/ai/culturax/2309/it/

TOKENIZER=sapienzanlp/Minerva-7B-base-v1.0

DEBUG=False
while true; do
    case "$1" in
        -d | --debug) DEBUG=True;;
        *) break ;;
    esac
    shift
done

if [ "$DEBUG" = "True" ]; then
    echo "Debug mode enabled"
    python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('$TOKENIZER')"
    exit 0
fi


OUTPUT_DIR=/leonardo_scratch/large/userexternal/lcolosi0/minerva/data/processed/it/culturax
rm -rf $OUTPUT_DIR # to be on the safe side
mkdir -p $OUTPUT_DIR # to be on the safe side

echo "Processing data"
echo "Saving to $OUTPUT_DIR"

python $BASE_DIR/scripts/data_prep/convert_dataset_hf.py \
    --dataset "$DATA" \
    --out_root "$OUTPUT_DIR" \
    --concat_tokens 2048 \
    --split train \
    --tokenizer "$TOKENIZER" \
    --no_wrap \
    --max_tokens 1_000_000_000_000 #equivalent to 2000 steps # omitt to process whole dataset


# python scripts/data_prep/convert_dataset_json.py \
#     --path "$DATA" \
#     --out_root "$OUTPUT_DIR" \
#     --split train \
#     --concat_tokens 4096 --tokenizer $SCRATCH/models/minerva-7B-base \
#     --max_tokens 100_000_000_000_000
