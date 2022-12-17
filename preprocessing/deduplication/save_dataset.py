from datasets import load_dataset

dataset = load_dataset("oscar-corpus/OSCAR-2201")
dataset.save_to_disk()
