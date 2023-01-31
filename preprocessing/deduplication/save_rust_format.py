# import csv 
import pandas as pd
import string

from datasets import load_dataset
from tqdm import tqdm


def find_whitespace(text):
    for i, c in enumerate(text):
        if c in string.whitespace:
            yield i


oscar_small = pd.DataFrame(load_dataset("ola13/small-oscar")["train"][:10])
query_length = 100

with open("queries.bin", "wb") as f:
    for idx, line in tqdm(oscar_small.iterrows()):
        text = line["text"]
        whitespace_idx = [-1] + list(find_whitespace(text))

        for i in whitespace_idx[::2]:
            if i + query_length + 1 < len(text):
                query = text[(i + 1) : (i + query_length + 1)]
                query_in_bytes = query.encode("utf-8")
                size = len(query_in_bytes)
                bytes_representation = size.to_bytes(4, "little")
                f.write(bytes_representation)
                f.write(query_in_bytes)