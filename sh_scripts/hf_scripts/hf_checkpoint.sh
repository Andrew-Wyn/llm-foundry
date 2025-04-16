#! /bin/bash

# Set up directories paths
BASE_DIR=/leonardo/home/userexternal/$(whoami)/minerva/llm-foundry
PYENV="$BASE_DIR"/llmfoundry-venv
DATA_DIR=/leonardo_scratch/large/userexternal/$(whoami)/minerva/replica
RUNS_DIR="$DATA_DIR"/runs
HF_CPT_DIR="$DATA_DIR"/huggingface-cpt

# Optionally save the new checkpoint to the hub
PUSH=False
while true; do
    case "$1" in
        -p | --push) PUSH=True;;
        *) break ;;
    esac
    shift
done

# Extract the latest checkpoint from the composer checkpoint
latest_dir=$(ls -lt $RUNS_DIR --group-directories-first | grep "^d" | head -n 1 | awk '{print $NF}')
run_name="$latest_dir"
rank=0 #$2

latest_file=$(ls -l "$RUNS_DIR"/"$run_name"/ | sort -V -k 9 | grep -o 'ep0-ba[0-9]*-rank0.pt' | tail -n 1)
base_name=$(echo "$latest_file" | sed 's/-rank0\.pt$//')

echo "Extracting checkpoint:" "$base_name" from "$RUNS_DIR"/"$run_name"

if [ -z "$run_name" ]; then
    echo "run_name is not provided"
    exit 1
fi

source "$PYENV"/bin/activate

python "$BASE_DIR"/scripts/inference/convert_composer_to_hf.py \
    --composer_path "$RUNS_DIR"/"$run_name"/latest-rank"$rank".pt \
    --hf_output_path "$HF_CPT_DIR"/"$run_name"/huggingface/"$base_name"

echo "Checkpoint saved at" "$HF_CPT_DIR"/"$run_name"/huggingface/"$base_name"

if [ "$PUSH" = "True" ]; then
    echo "Pushing checkpoint to the hub"
    bash "$BASE_DIR"/sh_scripts/hf_scripts/push_to_hub.sh \
        --run_name "$run_name" \
        --branch_name "$base_name"
fi
