#! /bin/bash

# Set up directories paths
BASE_DIR=/leonardo/home/userexternal/$(whoami)/minerva/llm-foundry

PYENV="$BASE_DIR"/llmfoundry-venv
source "$PYENV"/bin/activate

DATA_DIR=/leonardo_scratch/large/userexternal/$(whoami)/minerva/replica
export HF_HOME="$DATA_DIR"/.cache/hf_home/
HF_CPT_DIR="$DATA_DIR"/huggingface-cpt


TOKENIZER="sapienzanlp/Minerva-7B-base-v1.0"
# TOKENIZER=/leonardo_scratch/large/userexternal/lcolosi0/.cache/hf_home/hub/foo
# TOKENIZER="/leonardo_scratch/large/userexternal/$(whoami)/minerva-pretraining/tokenizer"
# TOKENIZER="/leonardo_scratch/large/userexternal/lcolosi0/minerva-pretraining/tokenizer"
# DATASET="/leonardo/home/userexternal/lcolosi0/test-tokenizer-data/train"
# DATASET="/leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/it/culturax-filtered/train"
# DATASET="/leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/it/culturax-filtered/train/"
DATASET="/leonardo_scratch/fast/mnrv_bblscp/minerva_pretraining/data/processed/it/culturax/train/"
# DATASET=/leonardo/home/userexternal/lcolosi0/test-tokenizer-data/train/

echo "Running read_mds.py"
python  $BASE_DIR/tests/tokenizers/experiments/read_mds.py \
    --tokenizer $TOKENIZER \
    --branch main \
    --max_pairs 10 \
    --dataset $DATASET


    # --dataset /leonardo_scratch/fast/IscrB_medit/training/minerva-7B-900B_it-900B_en-200B-code-21052024/data/processed/it/culturax-filtered/train \
    # --tokenizer sapienzanlp/Minerva-7B-base-v1.0 \

# DATASET_LIST=(
#     /leonardo/prod/data/ai/culturax/2309/it/
# )
# EXTENDED_LANG_LIST=(
#     italian
#     # english
# )
# LANG_LIST=(
#     it
#     # en
# )

# for i in "${!EXTENDED_LANG_LIST[@]}"; do
#     DATASET="${DATASET_LIST[$i]}"

#     for j in "${!EXTENDED_LANG_LIST[@]}"; do
#         EXTENDED_LANG="${EXTENDED_LANG_LIST[$j]}"
#         LANG="${LANG_LIST[$j]}"
#         echo "→ Dataset: $DATASET | Language: $LANG"

#         python "$SCRIPT_DIR/../tests/tokenizers/fertility_compute.py" \
#             --tokenizer "$TOKENIZER" \
#             --dataset "$DATASET" \
#             --ext_lang "$EXTENDED_LANG" \
#             --lang "$LANG" \
#             --local
#     done
# done
# python "$SCRIPT_DIR"/../tests/tokenizers/test_integrity.py \
#     --dataset CULTURAX \
#     --ext_lang it \
#     --lang default \
#     --n_samples 10 \
#     --show-full-text \
#     --local /leonardo/prod/data/ai/culturax/2309/it/ \
#     --dump
#
