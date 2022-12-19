import argparse
from os import cpu_count
from typing import List, Dict

from datasets import load_dataset, Dataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True,
                        help="Path to the dataset you're using on the HF hub. Pass e.g. `csv` or `json` and `data_files=path_on_disk` to load something locally")
    parser.add_argument('--subset', type=str, default=None, help="Subset of the dataset you're using, if needed")
    parser.add_argument('--data_files', type=str, default=None, help="Path to the dataset on disk if using local files")
    parser.add_argument('--ratio', type=float, help="Subsampling ratio", required=True)
    parser.add_argument('--pre_shuffle', action="store_true", help="Whether to shuffle the dataset in advance")
    return parser.parse_args()


def get_size_per_example(texts: List[str]) -> Dict:
    size_values = [len(text.encode()) for text in texts]
    examples = {"bytes_len": size_values}
    return examples


def linear_search_with_best_guess(dataset, ratio):
    """
    :param dataset: The HF dataset to split
    :param ratio: The ratio of bytes we want to keep
    :return: The cutoff point, rounded down
    """
    total_size = sum(dataset["bytes_len"])
    target_size = total_size * ratio
    current_index = int(round(len(dataset) * ratio))
    current_size = sum(dataset.select(range(current_index))["bytes_len"])

    if current_size > target_size:
        current_index -= 1
        while True:
            current_size -= dataset[current_index]["bytes_len"]
            if current_size < target_size:
                break
            current_index -= 1

    else:
        while True:
            current_size += dataset[current_index]["bytes_len"]
            if current_size > target_size:
                break
            current_index += 1

    below = sum(dataset.select(range(current_index))["bytes_len"])
    above = below + dataset[current_index]["bytes_len"]
    assert below < target_size < above, f"Sizes {below}, {target_size}, {above} not ordered, bad splitting"

    return current_index


def output_path(args):
    if args.data_files is not None:
        # assumes there's an extension
        path = args.data_files.split(".")[:-1]
        path += f"_{args.ratio}_subsample"
        path += ".jsonl"
    else:
        path = f"{args.name}_{args.subset}_{args.ratio}_subsample.jsonl"
    return path


if __name__ == "__main__":
    args = get_args()
    dataset = load_dataset(args.name, args.subset, data_files=args.data_files, num_proc=cpu_count(), split="train")
    dataset = dataset.map(
        get_size_per_example,
        batched=True,
        num_proc=cpu_count(),
        batch_size=1024,
        input_columns=["text"],
    )
    if args.pre_shuffle:
        dataset = dataset.shuffle()
    cutoff_point = linear_search_with_best_guess(dataset, args.ratio)
    dataset = dataset.select(range(cutoff_point))
    dataset.to_json(output_path(args), num_proc=cpu_count())
