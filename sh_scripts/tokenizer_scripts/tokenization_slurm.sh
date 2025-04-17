#!/bin/bash

#SBATCH --job-name=tokenization                         # Job name
#SBATCH --output=/leonardo/home/userexternal/lcolosi0/minerva-pretraining/logs/data-processing/%x-%j.out
#SBATCH --error=/leonardo/home/userexternal/lcolosi0/minerva-pretraining/logs/data-processing/%x-%j.err

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
export HF_HOME="$DATA_DIR"/.cache/hf_home/

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd $BASE_DIR

# DATA_ARROW=/leonardo/prod/data/ai/culturax/2309/it/
DATA_ARROW=/leonardo/prod/data/ai/culturax/2309/en/

# TOKENIZER=/leonardo_scratch/large/userexternal/$(whoami)/minerva-pretraining/tokenizer
# TOKENIZER=/leonardo_scratch/large/userexternal/$(whoami)/minerva-pretraining/tokenizer-32768
# TOKENIZER=/leonardo_scratch/large/userexternal/lcolosi0/.cache/hf_home/hub/foo
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

# OUTPUT_DIR=/leonardo_scratch/large/userexternal/$(whoami)/minerva-pretraining/data/processed
# OUTPUT_DIR=/leonardo_scratch/fast/mnrv_bblscp/minerva_pretraining/data/processed/it/culturax
OUTPUT_DIR=/leonardo_scratch/fast/mnrv_bblscp/minerva_pretraining/data/processed/en/culturax
rm -rf $OUTPUT_DIR # to be on the safe side
mkdir -p $OUTPUT_DIR # to be on the safe side

echo "Processing data"
echo "Saving to $OUTPUT_DIR"


# --path "$DATA_ARROW" \
python $BASE_DIR/scripts/data_prep/convert_dataset_hf.py \
    --dataset "$DATA_ARROW" \
    --out_root "$OUTPUT_DIR" \
    --concat_tokens 2048 \
    --split train \
    --tokenizer "$TOKENIZER" \
    --no_wrap \
    # --max_tokens 2048 #equivalent to 2000 steps # omitt to process whole dataset

    # --max_samples 273_448_960_000
    # --compression zstd \
    # --bos_text "<s>" \
    # --eos_text "</s>" \


# List of dataset paths:
#
RedPajama-v2: /leonardo/prod/data/ai/red pajama/2.0
• CulturaX: /leonardo/prod/data/ai/culturax/2309
• Wikipedia: wikimedia/wikipedia (*)
• Gutenberg (IT): /leonardo scratch/fast/IscrB medit/data/books it/gutenberg.jsonl
• Wikisource (IT):
/leonardo scratch/fast/IscrB medit/data/books it/wikisource it books clean.jsonl
EurLex: NLP-AUEB/eurlex (*)
Gazzetta Ufficiale: /leonardo scratch/fast/IscrB medit/data/gazetta/gazetta.jsonl
FineWeb: HuggingFaceFW/fineweb (*)
ArXiv: /leonardo scratch/fast/IscrB medit/data/arxiv
Gutenberg (EN): /leonardo scratch/fast/IscrB medit/data/books en/gutenberg.jsonl
StackExchange: /leonardo/prod/data/ai/red pajama/2.0
The StackV2: /leonardo scratch/fast/IscrB medit/data/starcoder2 smol batches
