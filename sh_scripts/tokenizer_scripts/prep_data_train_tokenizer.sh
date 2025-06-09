#!/bin/bash

export BASE_DIR=/leonardo/home/userexternal/$(whoami)/minerva/llm-foundry

export PYENV="$BASE_DIR"/llmfoundry-venv
source "$PYENV"/bin/activate

export HF_HOME=/leonardo_scratch/large/userexternal/$(whoami)/.cache/hf_home/

OUTPUT_DIR=/leonardo_scratch/large/userexternal/$(whoami)/minerva/data
OUTPUT_DATA_DIR="$OUTPUT_DIR"/tokenizer-data
mkdir -p $OUTPUT_DATA_DIR

DATA_ARROW_LIST=(
    /leonardo/prod/data/ai/culturax/2309/it/
    /leonardo/prod/data/ai/culturax/2309/en/
)

echo "Extracting data for tokenizer"
for DATA_ARROW in "${DATA_ARROW_LIST[@]}"; do
    echo "Dataset: $DATA_ARROW"
    dataset_name=$(basename "${DATA_ARROW%/}")
    # out_file="$OUTPUT_DATA_DIR/tokenizer-data-${dataset_name}.txt"
    python "$BASE_DIR/scripts/data_prep/extract_txt_for_tokenizer.py" \
        --dataset "$DATA_ARROW" \
        --output-file "$OUTPUT_DATA_DIR"/tokenizer-data.txt \
        --max-samples 2000000 \
        --shuffle \
        --streaming
done
