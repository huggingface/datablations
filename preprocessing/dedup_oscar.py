from datasets import load_from_disk
import string

def find_whitespace(text):
    for i, c in enumerate(text):
        if c in string.whitespace:
            yield i

def 

def get_segmentation(text, passage_tokens, overlap_tokens):
    whitespace_idx = [-1] + list(find_whitespace(text))
    unique_tokens = passage_tokens - overlap_tokens
    passages = []
    for i in range(0, len(whitespace_idx), unique_tokens):
        if i + passage_tokens >= len(whitespace_idx):
            passages.append((whitespace_idx[i] + 1, len(text)))
            break
        passages.append((whitespace_idx[i] + 1, whitespace_idx[i + passage_tokens] + 1))
    return passages

if __name__ == "__main__":
    oscar = load_from_disk("/home/piktus_huggingface_co/lumi/preprocessed_data/oscar_025")["train"]

    with open("/home/piktus_huggingface_co/lumi/preprocessed_data/oscar_025/queries.txt", "w") as queries:
    for line in oscar:
        text = line["text"]
        whitespace_idx = [-1] + list(find_whitespace(text))

        for i in whitespace_idx:
            if i + 101 < len(text):
                queries.write(text[i+1:i+101] + "\n")

