import argparse
import ast
import glob
import os

from datasets import DatasetDict, concatenate_datasets, load_from_disk


def get_perplexity(meta):
    meta = ast.literal_eval(meta) if isinstance(meta, str) else meta
    perplexity_score = meta["perplexity_score"]
    return float(perplexity_score)


if __name__ == "__main__":
    """
    Export your huggingface token which gives access to the `bigscience-catalogue-lm-data` organization.
    Run the following in the terminal where you execute the code (replace the XXX with your actual token):
    ```
    export HUGGINGFACE_TOKEN=hf_XXXXXXXXXXXXXXX
    ```
    """
    HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
    if HUGGINGFACE_TOKEN is None:
        raise RuntimeError("Hugging Face token not specified.")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_dir",
        type=str,
        default="/mnt/disks/looking_glass_storage/data/perplexity_filtered/lumi/",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100000,
    )
    args = parser.parse_args()

    files = glob.glob(args.base_dir + "roots_en_*")
    dataset_names = [f.split("/")[-1] for f in files]
    print(dataset_names)

    datasets = []
    for dataset_name in dataset_names:
        print("Processing", dataset_name)
        ds = load_from_disk(args.base_dir + dataset_name)
        ds = ds["train"]
        updated_dataset = ds.map(
            lambda example: {
                "text": example["text"],
                "meta": {"perplexity_score": get_perplexity(example["meta"])},
            },
            num_proc=48,
        )
        datasets.append(updated_dataset)

    roots_en = concatenate_datasets(datasets)
    roots_en.shuffle()
    small_roots_en = DatasetDict({"train": roots_en.select(range(args.sample_size))})
    small_roots_en.push_to_hub("ola13/small-roots_en", private=True, token=HUGGINGFACE_TOKEN)
    print("Pushed roots_en to hub.")
