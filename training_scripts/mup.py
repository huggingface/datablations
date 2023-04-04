"""
muP Preparation from https://github.com/microsoft/mutransformers#basic-usage-of-models

!git clone https://github.com/microsoft/mutransformers.git
%cd mutransformers
!pip install -r requirements.txt
!pip install -e .
!pip install -q datasets
"""

from mutransformers import GPT2Config, GPT2LMHeadModel
from mup import make_base_shapes, set_base_shapes, MuAdamW
from functools import partial

# define a base model
base_config = GPT2Config(
    hidden_size=256,
    intermediate_size=256,
    num_attention_heads=16,
)
base_model = GPT2LMHeadModel(config=base_config)
# define a delta models where we vary all "widths" we want to vary
delta_config = GPT2Config(
    hidden_size=200,
    intermediate_size=300,
    num_attention_heads=5,
)
delta_model = GPT2LMHeadModel(config=delta_config)
# define a base shape object based on comparing delta_model against base_model
base_shapes = make_base_shapes(base_model, delta_model, savefile='gpt256.bsh')

# define target model
target_config = GPT2Config(
    hidden_size=1024,
    intermediate_size=1024*4,
    num_attention_heads=32,
)
target_model = GPT2LMHeadModel(config=target_config)

# set base shapes
set_base_shapes(target_model, base_shapes)
# you can alternatively load base shape from file
# set_base_shapes(target_model, 'bert256.bsh')

# re-initialize
target_model.apply(target_model._init_weights)

# make sure to use mup optimizers for training
optimizer = MuAdamW(target_model.parameters(), lr=1e-3)


"""
Training code
Train billion parameter models on 100M tokens of C4
Adapted from:
https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb
"""

# mup-params-tokens
model_name = "mup-200m-100m"

from datasets import load_dataset
datasets = load_dataset('datablations/c4-100m')
!wget https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz
val_dataset = load_dataset('json', data_files='c4-validation.00000-of-00008.json.gz')['train']
# val_dataset = load_dataset('c4', 'en', split='validation[:10%]')
datasets["validation"] = val_dataset

datasets = datasets.select_columns("text")

from transformers import AutoTokenizer, Trainer, TrainingArguments
tokenizer = AutoTokenizer.from_pretrained("gpt2")

tokenized_datasets = datasets.map(lambda x: tokenizer(x["text"]), batched=True, num_proc=4, remove_columns=["text"])

block_size = tokenizer.model_max_length
# block_size = 128

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
    total_length = (total_length // block_size) * block_size
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result

lm_datasets = tokenized_datasets.map(
    group_texts,
    batched=True,
    batch_size=1000,
    num_proc=4,
)

training_args = TrainingArguments(
    model_name,
    evaluation_strategy = "epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=True,
)

trainer = Trainer(
    model=target_model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    optimizers=(optimizer, None), # Use mup optimizer & default scheduler
)

trainer.train()


import math
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.push_to_hub()