from datasets import load_dataset
import argparse
from text_dedup.exact_dedup import PythonSuffixArrayDeduplicator, GoogleSuffixArrayDeduplicator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, help="Path to the dataset you're using on the HF hub. Pass e.g. `csv` or `json` and `data_files=path_on_disk` to load something locally")
    parser.add_argument('--subset', type=str, default=None, help="Subset of the dataset you're using, if needed")
    parser.add_argument('--data_files', type=str, default=None, help="Path to the dataset on disk if using local files")
    parser.add_argument('--path_on_disk', type=str, required=True, help="Path to the Rust dedup implem on your disk, see https://github.com/google-research/deduplicate-text-datasets")
    parser.add_argument('--cache_dir', type=str, required=True, help="Where all the suffix tree files will get built")
    return parser.parse_args()


def generator_from_dataset(dataset):
    for item in dataset:
        yield item["text"]


if __name__ == "__main__":
    args = get_args()
    dataset = load_dataset(args.name, args.subset, data_files=args.data_files, use_auth_token=True, split="train")
    corpus = generator_from_dataset(dataset)

    deduplicator = GoogleSuffixArrayDeduplicator(k=10, merge_strategy='overlapping', google_repo_path=args.path_on_disk, cache_dir=args.cache_dir)
    slices = deduplicator.fit_predict(corpus)
    for sentence, intervals in zip(corpus, slices):
        print(sentence)
        print([sentence.encode('utf-8')[s].decode('utf-8', errors='ignore') for s in intervals])
