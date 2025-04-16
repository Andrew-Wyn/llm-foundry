import os
import nltk
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

# Set path of current script
CURR_PATH = os.path.dirname(os.path.realpath(__file__))
# TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"
# TOKENIZER_NAME = "google/gemma-7b"
# TOKENIZER_NAME = "sapienzanlp/minestral-1B-100B_it-100B_en-cx-04032024"
# TOKENIZER_NAME = "meta-llama/Meta-Llama-3-8B"
# TOKENIZER_NAME = "/home/luca/llm-cva-tatent/adaptation/mistral_7B-adapted_3B_sgd_8k_substitution"
# TOKENIZER_NAME = "sapienzanlp/Minerva-7B-base-v1.0"
TOKENIZER_NAME = "/mnt/disk1/nfs/scire-slurm/train-tokenizer/tokenizer" #"BS-team/minerva-replica-tokenizer"
LIMIT_DOCS = 50_000


# LANG = "it"
# LANG = "en"
# # LANG_LONG = "italian"
# LANG_LONG = "english"
# # DS = "CULTURAX"
# DS = "WIKIPEDIA"

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ext_lang",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        default=None,
    )

    args = parser.parse_args(args)
    return args

def main():

    args = parse_args()
    LANG = args.lang
    LANG_LONG = args.ext_lang
    DS = args.dataset
    LOCAL = args.local

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    print(f"Using tokenizer: {TOKENIZER_NAME} on dataset: {DS} in language: {LANG_LONG}")

    if DS == "CULTURAX":
        if LOCAL:
            DS_STREAM = load_dataset(DS, LANG, split='train', streaming=True)
        else:
            DS_STREAM = load_dataset('uonlp/CulturaX', LANG, split='train', streaming=True)
    else:
        DS_STREAM = load_dataset("wikimedia/wikipedia", f"20231101.{LANG}", split='train', streaming=True)

    fertility = 0

    for it, sample in enumerate(tqdm(DS_STREAM)):
        if it == LIMIT_DOCS:
            break

        if it > 0 and it % 10_000 == 0:
            print(f"Runtime fertility: {fertility / it}")

        sample_text = sample["text"]

        words = word_tokenize(sample_text, language=LANG_LONG)
        tokens_id = tokenizer(sample_text, add_special_tokens=False).input_ids

        sample_fertility = len(tokens_id) / len(words)
        fertility += sample_fertility

    fertility = fertility / LIMIT_DOCS

    print(f"{TOKENIZER_NAME} - FERTILITY over {LIMIT_DOCS} over {DS} {LANG_LONG}: {fertility}")

    output_file = os.path.join(CURR_PATH, f"../output/fertility_{TOKENIZER_NAME.replace('/', '_')}_{DS}_{LANG}.txt")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        mode = "a"
    else:
        mode = "w"
    with open(output_file, mode) as f:
        f.write(f"{TOKENIZER_NAME} - FERTILITY over {LIMIT_DOCS} over {DS} {LANG_LONG}: {fertility}")


if __name__ == "__main__":
    main()
    # BS-team/minerva-replica-tokenizer - FERTILITY over 50000 over WIKIPEDIA italian: 1.7817975814109819
# BS-team/minerva-replica-tokenizer - FERTILITY over 50000 over WIKIPEDIA english: 2.2652814444988936
