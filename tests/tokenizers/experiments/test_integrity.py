import os
import random
import argparse
import json
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

CURR_PATH = os.path.dirname(os.path.realpath(__file__))

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name: 'CULTURAX' or 'WIKIPEDIA'")
    parser.add_argument("--ext_lang", type=str, required=True, help="Full language name (e.g., 'english')")
    parser.add_argument("--lang", type=str, required=True, help="Short lang code (e.g., 'en')")
    parser.add_argument("--n_samples", type=int, default=5, help="Number of random samples to inspect")
    parser.add_argument("--show-full-text", action="store_true", help="Print full text instead of truncating")
    parser.add_argument("--local", type=str, required=False, default="", help="use loacl dataset (always hf format)")
    parser.add_argument("--dump", action="store_true", help="Optional output file path")
    return parser.parse_args(args)

def main():
    args = parse_args()

    TOKENIZER_NAME = "sapienzanlp/Minerva-7B-base-v1.0"
    print(f"🔤 Loading tokenizer from: {TOKENIZER_NAME}")
    # tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision="step90000")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, revision="main")

    print(f"📚 Streaming dataset: {args.dataset} [{args.lang}]")

    # Load streaming dataset
    if args.dataset.upper() == "CULTURAX":
        if args.local:
            dataset = load_dataset(args.local, args.lang, split="train", streaming=True)
        else:
            dataset = load_dataset("uonlp/CulturaX", args.lang, split="train", streaming=True)
    else:
        dataset = load_dataset("wikimedia/wikipedia", f"20231101.{args.lang}", split="train", streaming=True)

    # Reservoir sampling
    reservoir = []
    seen = 0
    for sample in tqdm(dataset, desc="Streaming samples..."):
        seen += 1
        if len(reservoir) < args.n_samples:
            reservoir.append(sample)
        else:
            j = random.randint(0, seen - 1)
            if j < args.n_samples:
                reservoir[j] = sample

        if seen > args.n_samples * 100:
            # Bail out after a lot of samples if desired (optional)
            break

    print(f"✅ Collected {len(reservoir)} random samples.\n")

    results = []
    dump_list = []
    for i, sample in enumerate(reservoir):
        text = sample["text"]

        encoded = tokenizer(text, add_special_tokens=False)
        token_ids = encoded.input_ids
        decoded = tokenizer.decode(token_ids)
        tokens = tokenizer.convert_ids_to_tokens(token_ids)

        preview = {
            "sample_id": i + 1,
            "original_text": text if args.show_full_text else text[:300],
            # "token_ids": token_ids,
            "tokens": tokens,
            "decoded_text": decoded if args.show_full_text else decoded[:300],
        }

        results.append(preview)
        dump_list.append({
            f"original_text_{i}": text
        })

        print(f"=== Sample #{i + 1} ===")
        print(json.dumps(preview, indent=2, ensure_ascii=False))
        print("=" * 60 + "\n")


    # Optionally dump to file
    output_file = os.path.join(CURR_PATH, f"outputs/integrity/{TOKENIZER_NAME}_{args.dataset}_{args.lang}.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    if args.dump:
        dump_file = os.path.join(CURR_PATH, f"tests-outputs/dump_{TOKENIZER_NAME}_{args.dataset}_{args.lang}.json")
        os.makedirs(os.path.dirname(dump_file), exist_ok=True)
        print(f"💾 Dumping output to: {dump_file}")
        with open(dump_file, "w", encoding="utf-8") as f:
            for dump_dict in dump_list:
                json.dump(dump_dict, f, indent=2, ensure_ascii=False)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"💾 Saved output to: {output_file}")

if __name__ == "__main__":
    main()
