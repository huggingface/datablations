import argparse
import os
from typing import List, Dict
import subprocess
import shlex

import numpy as np
import pyarrow as pa
from datasets import load_dataset, Dataset, concatenate_datasets
from tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                        help="Path to the dataset you're using on the HF hub. Pass e.g. `csv` or `json` and `data_files=path_on_disk` to load something locally")
    parser.add_argument('--subset', type=str, default=None, help="Subset of the dataset you're using, if needed")
    parser.add_argument('--data_files', type=str, default=None, help="Path to the dataset on disk if using local files")
    parser.add_argument('--ratios', nargs='+', type=float, help="Subsampling ratios", required=True)
    parser.add_argument('--names', nargs='+', type=str, help="Names for the produced subsets", required=False)
    parser.add_argument('--pre_shuffle', action="store_true", help="Whether to shuffle the dataset in advance")
    parser.add_argument('--shuffle_seed', type=int, default=0, help="Shuffling seed")
    return parser.parse_args()


def get_size_per_example(texts: List[str]) -> Dict:
    size_values = [len(text.encode()) for text in texts]
    examples = {"bytes_len": size_values}
    return examples


def get_total_byte_size(dataset):
    return pa.compute.sum(dataset.data["bytes_len"]).as_py()


def output_path(args, ratio, name):
    if name is None:
        name = f"{ratio}_subsample"
    if args.data_files is not None:
        # assumes there's an extension
        path = args.data_files.split(".")[:-1]
        path += f"_{name}"
        path += ".jsonl"
    else:
        path = f"{args.name}_{args.subset}_{name}.jsonl"
    return os.path.abspath(path)


if __name__ == "__main__":
    args = get_args()

    if args.names is None:
        args.names = [None] * len(args.ratios)
    else:
        assert len(args.names) == len(args.ratios)

    dataset = load_dataset(args.name, args.subset, data_files=args.data_files, num_proc=os.cpu_count(), split="train")

    dataset = dataset.map(
        get_size_per_example,
        batched=True,
        num_proc=os.cpu_count(),
        batch_size=1024,
        input_columns=["text"],
    )

    if args.pre_shuffle:
        # this is going to be incredibly slow on large datasets
        dataset = dataset.shuffle(args.shuffle_seed)
        dataset = dataset.flatten_indices(num_proc=os.cpu_count())

    cumsum_sizes = pa.compute.cumulative_sum(dataset.data["bytes_len"])
    cumsum_ds = Dataset(pa.Table.from_arrays([cumsum_sizes], names=["cumsum_sizes"]))
    dataset = concatenate_datasets([dataset, cumsum_ds], axis=1)
    total_size = dataset[-1]["cumsum_sizes"]

    dataset = dataset.with_format("numpy")
    ratios_and_names = sorted(list(zip(args.ratios, args.names)), key=lambda x: x[0], reverse=True)
    base_file = args.data_files
    assert dataset._indices is None

    for ratio, name in tqdm(ratios_and_names):
        cutoff_point = np.searchsorted(dataset["cumsum_sizes"], total_size * ratio)
        if base_file is None:
            subset = dataset.select(range(cutoff_point)).remove_columns(["bytes_len", "cumsum_sizes"])
            assert subset._indices is None
            subset.to_json(output_path(args, ratio, name), num_proc=64, batch_size=100_000)
            base_file = output_path(args, ratio, name)
        else:
            subprocess.run(shlex.split(f"head -{cutoff_point} {base_file}"),
                           stdout=open(output_path(args, ratio, name), "w"), check=True)
