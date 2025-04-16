import argparse
from glob import glob
import os
import sys
from pathlib import Path
from typing import Optional, Union

import datasets as hf_datasets
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

def build_hf_dataset(
    dataset_name: str,
    split: str,
    data_type: str = "json",
    data_subset: Union[str, None] = None,
    streaming: bool = True,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: Optional[int] = None,
) -> hf_datasets.Dataset:
    """Load HF dataset from disk or Hub"""
    is_local = os.path.exists(dataset_name)

    if is_local:
        if os.path.isdir(dataset_name) and os.path.exists(os.path.join(dataset_name, "dataset_dict.json")):
            print(f"🔹 Loading dataset from disk (saved HF format): {dataset_name}")
            if streaming:
                print("⚠️  Streaming not supported with `load_from_disk()`. Disabling streaming.")
            dataset = load_from_disk(dataset_name)
            if split in dataset:
                dataset = dataset[split]
            else:
                raise ValueError(f"Split '{split}' not found in saved dataset at {dataset_name}")
        else:
            print(f"🔹 Loading dataset from files: {dataset_name}/*.{data_type}")
            data_files = glob(f'{dataset_name}/*.{data_type}')
            dataset = load_dataset(
                data_type,
                data_files=data_files,
                split=split,
                streaming=streaming,
                num_proc=num_workers if not streaming else None,
            )
    else:
        print(f"🔹 Loading dataset from Hugging Face Hub: {dataset_name}")
        dataset = load_dataset(
            path=dataset_name,
            name=data_subset,
            split=split,
            streaming=streaming
        )

    # if shuffle:
    #     print("🔀 Shuffling dataset")
    #     dataset = dataset.shuffle(seed=seed)

    return dataset

def main(dataset, output_file, max_samples, data_type, streaming=False, shuffle=False):
    hf_dataset = build_hf_dataset(dataset, "train", data_type, num_workers=4, streaming=streaming, shuffle=shuffle)

    output_file = Path(output_file)
    if os.path.exists(output_file):
        print(f"Output file already exists, appending to: {output_file}")
        mode = "a"
    else:
        print(f"Writing to: {output_file}")
        mode = "w"
    output_file.parent.mkdir(parents=True, exist_ok=True)


    progress_bar = tqdm(total=max_samples)
    i = 0
    with open(output_file, mode) as f_out:
        for line in hf_dataset:
            if i >= max_samples:
                break
            if "text" in line:
                f_out.write(f"{line['text']}\n")
            else:
                f_out.write(str(line) + "\n")
            i += 1
            progress_bar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to local dataset folder or name of HF dataset.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to save the output .txt file.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Max number of rows to export.",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming mode for remote datasets.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Use for shuffling the dataset.",
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="json",
        help="Data type of the files (json, arrow, csv, etc).",
    )

    args = parser.parse_args()
    main(**vars(args))
