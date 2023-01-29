import os
from multiprocessing import cpu_count

from datasets import load_dataset

HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
print(HUGGINGFACE_TOKEN)
oscar = load_dataset(
    "oscar-corpus/OSCAR-2201", "en", use_auth_token=HUGGINGFACE_TOKEN, num_proc=cpu_count(), ignore_verifications=True
)

oscar.save_to_disk("/home/piktus_huggingface_co/lumi/oscar/")
