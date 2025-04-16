import argparse
import psutil
import sentencepiece as spm
from tqdm import tqdm
import transformers as tr
import time

from datasets import load_dataset
from transformers import LlamaTokenizerFast
from tokenizers import (
    pre_tokenizers,
    decoders,
    Tokenizer,
    normalizers,
)
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--tokenizer-folder", type=str)
    parser.add_argument("--vocab-size", type=int)
    parser.add_argument("--hf-class", type=str, default="LlamaTokenizerFast")
    parser.add_argument("--add-prefix-space", action="store_true")
    parser.add_argument("--byte-fallback", action="store_true")
    parser.add_argument("--push-to-hub", type=str, default=None)
    parser.add_argument("--use-bpe-hf-tokenizer", action="store_true")
    parser.add_argument("--debug", type=str, default=None)
    args = parser.parse_args()

    if args.use_bpe_hf_tokenizer:

        special_tokens = ["<s>", "</s>", "<unk>"]

        input_files = args.input.split(",")
        dataset = []
        print("Reading input files")
        for input_file in input_files:
            print(f"Reading {input_file}")
            with open(input_file, "r") as f:
                for line in tqdm(f):
                    dataset.append(line)

        def batch_iterator(dataset, batch_size=1000):
            for i in range(0, len(dataset), batch_size):
                yield dataset[i : i + batch_size]

        vocab_size = args.vocab_size

        # digits_pretokenizer = pre_tokenizers.Digits(individual_digits=False)
        bytelevel_pretokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=args.add_prefix_space, use_regex=True
        )

        bytelevel_decoder = decoders.ByteLevel(
            add_prefix_space=args.add_prefix_space, use_regex=True
        )

        tokenizer = Tokenizer(BPE(byte_fallback=args.byte_fallback))

        # tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        #     [digits_pretokenizer, bytelevel_pretokenizer]
        # )
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [bytelevel_pretokenizer]
        )
        tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
        tokenizer.decoder = bytelevel_decoder

        # 🧪 TEST pre-tokenization on a sample sentence
        print( bool(args.debug.capitalize()))
        if bool(args.debug.lower() == "true"):
            test_str = "In 1780, something happened."
            print("\nPre-tokenization test:")
            print(tokenizer.pre_tokenizer.pre_tokenize_str(test_str))
            exit()

        trainer = BpeTrainer(
            vocab_size=vocab_size, show_progress=True, special_tokens=special_tokens
        )

        tokenizer.train_from_iterator(batch_iterator(dataset), trainer=trainer)

        hf_class = getattr(tr, args.hf_class)
        tokenizer_wrapper = hf_class(
            tokenizer_object=tokenizer,
            vocab_size=vocab_size,
            # additional_special_tokens=SPECIAL_TOKENS,
            bos_token=special_tokens[0],
            eos_token=special_tokens[1],
            unk_token=special_tokens[2],
        )
        print(f"Saving HF tokenizer in {args.tokenizer_folder}")
        tokenizer_wrapper.save_pretrained(args.tokenizer_folder)
        if args.push_to_hub:
            tokenizer_wrapper.push_to_hub(args.push_to_hub)

    else:
        start = time.time()
        spm.SentencePieceTrainer.train(
            input=args.input,
            model_prefix=args.model_prefix,
            vocab_size=args.vocab_size,
            character_coverage=args.character_coverage,
            model_type="bpe",
            byte_fallback=True,
            split_digits=True,
            shuffle_input_sentence=True,
            num_threads=args.num_threads or psutil.cpu_count(logical=True),
        )
        end = time.time()
        print("Training took {} seconds".format(end - start))

        # assumes a llama tokenizer
        print("Saving HF tokenizer")
        hf_class = getattr(tr, args.hf_class)
        hf_tokenizer = hf_class(args.model_prefix + ".model")
        hf_tokenizer.save_pretrained(args.model_prefix + "-hf")
