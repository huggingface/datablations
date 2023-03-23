# Dedulplication experiments

For my current experiments I'm storing data on the [looking-glass-storage](https://console.cloud.google.com/compute/disksDetail/zones/europe-west1-b/disks/looking-glass-storage?project=huggingface-science-incubator) disk on GCP, mounted on the [looking-glass-cpu](https://console.cloud.google.com/compute/instancesDetail/zones/europe-west1-b/instances/looking-glass-cpu?project=huggingface-science-incubator) instance.

I'm following the process below.

### 1. CREATE SAMPLE DATASET.
Create a toy sample of the dataset using [`save_dataset_sample.py`](https://github.com/huggingface/datablations/blob/main/preprocessing/deduplication/save_dataset_sample.py) or [`save_roots.py`](https://github.com/huggingface/datablations/blob/main/preprocessing/deduplication/save_roots_sample.py). The (private) datasets are available under
- [ola13/small-c4](https://huggingface.co/datasets/ola13/small-c4) etc. for respective dataset names.

### 2. PREPROCESSING + PERPLEXITY SCORE.
We preprocess datasets by running the Big Science pipeline. Below is a sample command - you need to check out the [ola-lumi branch](https://github.com/bigscience-workshop/data-preparation/tree/ola-lumi). We calcuale perplexity agains a Wikipedia LM for each pre-processed datapoint and save in the dataset's metadata.

```
python preprocessing/training/01b_oscar_cleaning_and_filtering/main_filtering.py  --lang_dataset_id lumi --path_sentencepiece_model /mnt/disks/looking_glass_storage/lumi/kenlm/en.sp.model --path_kenlm_model /mnt/disks/looking_glass_storage/lumi/kenlm/en.arpa.bin --path_dir_save_dataset /mnt/disks/looking_glass_storage/lumi/preprocessed_data/ --dataset_name small-oscar --num_proc 104
```

### 3. INPUT FOR GOOGLE DEDUP.
We build input file for the [Google deduplication repo](https://github.com/google-research/deduplicate-text-datasets#collecting-the-duplicates-together) from a Hugging Face dataset using [`hf_dataset_to_file.py`](https://github.com/huggingface/datablations/blob/main/preprocessing/deduplication/hf_dataset_to_file.py). We're saving an extra file with a `.pos2id.pkl` suffix - for a position in the binary file constituting the input for deduplication, we save the id of the datapoint in the dataset on the hub. This way positions we get when using suffix arrays can be translated back to specific HF datapoints.

### 4. RUN GOODLE DEDUP.
Follow the dataset `README` to complete the steps below:
1. make_suffix_array
2. self-similar
3. collect

I'm experimenting with duplicated strings of length 50 and more.

### 4. DEDUPLICATE THE DATASET.
Currently the deduplication process is implemented in [this notebook](https://github.com/huggingface/datablations/blob/main/notebooks/dedup_investigation.ipynb) since we're only running on sample datasets for now. The notebook also contains some useful visualization functions.

Instead of actually deduplicating, we're adding metadata to each datapoint which can allow deduplication after performing further analysis. Specifically, for each textual datapoint we add the following additional info:
- `perplexity` - as calculated in point 1.
- `dup_ratio` - the fraction of the datapoint length which is duplicated. The value may go over 100 if dedected duplicated substrings are overlapping.
- `pairs` - pairs indicating boundaries of duplicated substrings - values are positions returned by the Google repo, so these are offsets in the file created by `hf_dataset_to_file.py`.
- `repetitions` - the actual duplicated substring (in bytes so it seems the hub interprets it as int) - there is an issue which messes with conversion from bytes to strings, I opened a relevant issue [here](https://github.com/google-research/deduplicate-text-datasets/issues/24)
- `cluster` - the list of lists of ids (the actual HF dataset ids) of datapoints which have substrings overlapping with those from the current sample.

Following the logic described above, I create datasets like the following:
- [ola13/small-c4-dedup](https://huggingface.co/datasets/ola13/small-c4-dedup)
- [ola13/small-c4-repetitions](https://huggingface.co/datasets/ola13/small-c4-repetitions) - a dataset of duplicated strings, their respective `pairs` and `ids` of HF dataset datapoints containing a given duplication.



# Sample multiquery query:
```
time ./target/debug/dedup_dataset count-occurrences-multi --data-file  /home/piktus_huggingface_co/lumi/dedup/oscar_025/oscar.train --query-file /home/piktus_huggingface_co/lumi/preprocessed_data/oscar_queries/queries_137974608.bin > test.out
```
