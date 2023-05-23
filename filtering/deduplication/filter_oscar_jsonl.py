import jsonlines
from collections import defaultdict
from datasets import load_from_disk, load_dataset, concatenate_datasets
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

oscar = load_from_disk("/home/piktus_huggingface_co/lumi/preprocessed_data/oscar-dedup-exapanded/")
# oscar = load_dataset("ola13/small-oscar")["train"]

oscar_shards = {}
for i in tqdm(range(160)):
    oscar_shards[i] = oscar.shard(num_shards=160, index=i)

def filter_shards(shard_id):
    print("Processing shard {}".format(shard_id))
    shard_lines = []
    for line in tqdm(oscar_shards[shard_id]):
        # if len(line["text"]) < 500:
        if (line["included_in_dedup"] and line["dup_ratio"] == 0.0) or ((not line["included_in_dedup"]) and (not line["has_dup_25"])):
            shard_lines.append({"text": line["text"]})
    return shard_lines


pool = Pool(160)
results = pool.map(filter_shards, [i for i in range(160)])
pool.close()
pool.join()

with jsonlines.open('/home/piktus_huggingface_co/lumi/preprocessed_data/oscar-dedup-25-exapanded.jsonl', mode='w') as writer:
    for shard_lines in results:
        writer.write_all(shard_lines)