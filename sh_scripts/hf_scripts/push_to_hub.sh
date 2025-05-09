#! /bin/bash


# Set up directories paths
BASE_DIR=/leonardo/home/userexternal/$(whoami)/minerva/llm-foundry
PYENV="$BASE_DIR"/llmfoundry-venv
DATA_DIR=/leonardo_scratch/large/userexternal/$(whoami)
export HF_HOME="$DATA_DIR"/.cache/hf_home/
HF_CPT_DIR="$DATA_DIR"/minerva/replica/huggingface-cpt

# Optionally push to main branch
run_name=
branch_name=
while true; do
    case "$1" in
        -r | --run_name) run_name="$2"; shift ;;
        -b | --branch_name) branch_name="$2"; shift ;;
        *) break ;;
    esac
    shift
done

if [ -z "$run_name" ]; then
    echo "run_name is not provided"
    exit 1
fi

if [ -z "$branch_name" ]; then
    echo "branch_name is not provided"
    exit 1
fi

source "$PYENV"/bin/activate
module load git-lfs/3.1.2 # enable git large file storage

echo Pushing checkpoint to the hub: Repo: BS-team/"$run_name", cpt: "$branch_name"
echo cpt path: "$HF_CPT_DIR"/"$run_name"/huggingface/"$branch_name"

python "$BASE_DIR"/scripts/inference/push_to_hub.py \
    --repo_id BS-team/"$run_name" \
    --folder_path "$HF_CPT_DIR"/"$run_name"/huggingface/"$branch_name" \
    --branch $branch_name
