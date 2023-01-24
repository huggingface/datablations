import argparse
import glob
import multiprocessing as mp
import os
import pickle
import random
import struct

import numpy as np
from datasets import load_from_disk
from tqdm import tqdm
from transformers import GPT2Tokenizer, T5Tokenizer

parser = argparse.ArgumentParser(description="Load a dataset.")
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--save_dir", type=str, required=True)
parser.add_argument("--source_dir", type=str, required=True)
parser.add_argument("--split", type=str, default="train")
parser.add_argument("--tokenize", action="store_true")
parser.add_argument("--tokenizer", type=str, default="gpt2")
parser.add_argument("--pre_sep", type=bytes, default=b"\xff\xff")
parser.add_argument("--post_sep", type=bytes, default=b"")
parser.add_argument("--sampling_rate", type=float, default=1.0)
args = parser.parse_args()


if args.tokenize:
    if args.tokenizer == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    elif args.tokenizer == "t5":
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
    else:
        raise

split = args.split
dataset_name = args.name
source_dir = args.source_dir + "/" + dataset_name
save_dir = args.save_dir + "/" + dataset_name + "_" + str(args.sampling_rate).replace(".", "")

print("Loading", dataset_name)
ds = load_from_disk(source_dir)
ds = ds[split]
print("Done loading")

print("Sampling rate", args.sampling_rate)

pre_sep = args.pre_sep
post_sep = args.post_sep

UID = 0


def sep():
    global UID
    UID += 1
    return pre_sep + struct.pack("<I", UID) + post_sep


def tok(x):
    if args.tokenize:
        out = tokenizer.encode(x.decode("utf8"))
        out = np.array(out, dtype=np.uint16).view(np.uint8).tobytes()
    else:
        out = x
    return out


if not os.path.exists(save_dir):
    os.mkdir(save_dir)

fout = open(os.path.join(save_dir, dataset_name + "." + split), "wb")

pos_to_id = {}

#with mp.get_context("fork").Pool(mp.cpu_count()) as p:
sizes = [0]

# IF ROOTS:
# for directory in glob.glob(source_dir + "*"):
#     print(directory)
#     ds = load_from_disk(directory)
#     ds = ds[split]

for i, b in tqdm(enumerate(ds)):
    if random.random() > args.sampling_rate:
        continue
    if b["text_length"] > 188944:
        continue
    next_line = sep() + b["text"].encode("utf8")
    fout.write(next_line)
    pos_to_id[sizes[-1]] = i
    sizes.append(sizes[-1] + len(next_line))

open(os.path.join(save_dir, dataset_name + "." + split + ".size"), "wb").write(
    np.array(sizes, dtype=np.uint64).tobytes()
)

with open(os.path.join(save_dir, dataset_name + "." + split + ".pos2id.pkl"), "wb") as f:
    pickle.dump(pos_to_id, f)
