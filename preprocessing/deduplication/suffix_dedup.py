from datasets import load_dataset
import argparse
from text_dedup.exact_dedup import PythonSuffixArrayDeduplicator, GoogleSuffixArrayDeduplicator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--subset', type=str, default=None)
    parser.add_argument('--path_on_disk', type=str, required=True)
    parser.add_argument('--cache_dir', type=str, required=True)
    return parser.parse_args()


def generator_from_dataset(dataset):
    for item in dataset:
        yield item["text"]


if __name__ == "__main__":
    args = get_args()
    dataset = load_dataset(args.name, args.subset, use_auth_token=True, split="train")
    corpus = generator_from_dataset(dataset)

    deduplicator = GoogleSuffixArrayDeduplicator(k=10, merge_strategy='overlapping', google_repo_path=args.path_on_disk, cache_dir=args.cache_dir)
    slices = deduplicator.fit_predict(corpus)
    for sentence, intervals in zip(corpus, slices):
        print(sentence)
        print([sentence.encode('utf-8')[s].decode('utf-8', errors='ignore') for s in intervals])
