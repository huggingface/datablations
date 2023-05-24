import argparse
import glob
import os
from functools import partial
from multiprocessing import Pool, cpu_count

from datasets import DatasetDict, load_from_disk


def save_dataset(dataset_name, base_dir, sample_size=100000, token=None):
    print("Processing", dataset_name)
    ds = load_from_disk(base_dir + dataset_name)
    ds.shuffle()
    while sample_size > len(ds["train"]):
        sample_size //= 10
    small_ds = DatasetDict({"train": ds["train"].select(range(sample_size))})
    small_ds.push_to_hub("ola13/small-" + dataset_name, private=True, token=token)
    print("Pushed", dataset_name, "to hub.")


if __name__ == "__main__":
    """
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
        required=True,
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=100000,
    )
    args = parser.parse_args()

    files = glob.glob(args.base_dir + "/*")
    datasets = [f.split("/")[-1] for f in files]
    print(datasets)

    workers = cpu_count()
    print("Number of workers:", workers)
    pool = Pool(workers)
    pool.map(
        partial(save_dataset, base_dir=args.base_dir, sample_size=args.sample_size, token=HUGGINGFACE_TOKEN),
        datasets,
    )
    pool.close()
    pool.join()
