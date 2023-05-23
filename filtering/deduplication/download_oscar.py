import os

from collections import Counter
from datasets import load_dataset

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

oscar = load_dataset(
    "oscar-corpus/OSCAR-2201",
    "en",
    use_auth_token=HUGGINGFACE_TOKEN,
    num_proc=128,
    ignore_verifications=True,
)
# oscar.save_to_disk("/home/piktus_huggingface_co/lumi/oscar/")

oscar_ids = oscar["train"]["id"]
print("Number of Oscar IDs", len(oscar_ids))

unique_ids = Counter(oscar_ids)
print(unique_ids.most_common(10))

