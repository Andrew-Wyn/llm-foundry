# from streaming import MDSReader
import os
import sys
import numpy as np
import argparse
from streaming import StreamingDataset
from transformers import AutoTokenizer

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Tokenize a dataset")
    parser.add_argument("--tokenizer", type=str, default="train", help="Tokenizer path")
    parser.add_argument("--branch", type=str, default="train", help="Tokenizer path")
    parser.add_argument("--dataset", type=str, default="train", help="Dataset to tokenize")
    parser.add_argument("--name", type=str, default="train", help="Dataset name")
    parser.add_argument("--max_pairs", type=int, default=1, help="Maximum number of pairs to tokenize")
    return parser.parse_args(args)

def main(args=None):
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # Load your tokenizer
    # TOKENIZER_NAME = "sapienzanlp/Minerva-350M-base-v1.0"
    # TOKENIZER_NAME = "sapienzanlp/Minerva-7B-base-v1.0"
    TOKENIZER_NAME = args.tokenizer #"/leonardo_scratch/large/userexternal/lcolosi0/minerva-pretraining/tokenizer"
    BRANCH_NAME = args.branch
    DATASET_PATH = args.dataset
    DATA_NAME = args.name
    MAX_PAIRS = args.max_pairs
    TAG = TOKENIZER_NAME.split("/")[-1]
    print(f"🔤 Loading tokenizer from: {TOKENIZER_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision=BRANCH_NAME)
    output_log_path = os.path.join(CURR_DIR, f"output/decoding/{TAG}_{BRANCH_NAME}_{DATA_NAME}.txt")
    os.makedirs(os.path.dirname(output_log_path), exist_ok=True)
    # Directory that contains your .mds shards
    # dataset_path = os.path.join(CURR_DIR, "output/train")
    # dataset_path = "/leonardo_scratch/large/userexternal/lcolosi0/minerva-pretraining/data/processed/train"
    dataset = StreamingDataset(
        local=DATASET_PATH,
        batch_size=1,     # ✅ REQUIRED NOW
        shuffle=False,
    )

    print(f"🔤 Tokenizing dataset: {DATASET_PATH}")
    print(f"🔤 Writing output to: {output_log_path}")
    # Open output file
    # Dump output
    with open(output_log_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(dataset):
            if i >= MAX_PAIRS:
                break

            if "text" in sample:
                original = sample["text"]
                if isinstance(original, bytes):
                    original = original.decode("utf-8", errors="replace")

                encoding = tokenizer(original, add_special_tokens=False, return_tokens=True)
                encoding = tokenizer(original, add_special_tokens=False, return_tokens=True)
                tokens = encoding  # This gives the list of token *strings*
                decoded = tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), skip_special_tokens=True)

                f.write(f"--- Sample {i+1} ---\n")
                f.write(f"[Original Text]:\n{original.strip()}\n")
                f.write(f"[Tokens]:\n{tokens}\n")
                f.write(f"[Decoded Text]:\n{decoded.strip()}\n\n")

                # Handle bytes token IDs
            elif "tokens" in sample:
                token_ids = sample["tokens"]

                # If token_ids are bytes, interpret them as int64
                if isinstance(token_ids, bytes):
                    try:
                        token_ids = np.frombuffer(token_ids, dtype="<q").tolist()  # little-endian int64
                    except Exception as e:
                        f.write(f"[Error decoding token_ids bytes as int64]: {e}\n")
                        continue

                # print(token_ids)
                tokens = tokenizer.convert_ids_to_tokens(token_ids)
                decoded = tokenizer.decode(token_ids, skip_special_tokens=True)

                f.write(f"--- Sample {i+1} ---\n")
                f.write(f"[Tokens]:\n{tokens}\n")
                f.write(f"[Decoded Text]:\n{decoded.strip()}\n\n")


if __name__ == "__main__":
    main()
