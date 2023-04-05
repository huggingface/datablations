"""
muP Preparation from https://github.com/microsoft/mutransformers#basic-usage-of-models

!git clone https://github.com/microsoft/mutransformers.git
%cd mutransformers
!pip install -r requirements.txt
!pip install -e .
!pip install -q datasets


With our CC-like architectures we found that
7m params & 100M tokens -> 8.1 loss
1b1 params & 100M tokens -> 6.6 loss
2b8 params & 100M tokens -> 7.5 loss
So looking to run the last two, which in our CC setup have the hyperparams:
(d_model ffw_size kv_size n_heads n_layers)
PARAM_1143M=(1792 7168 128 14 26)
PARAM_2980M=(2560 10240 128 20 34)

target_config -> base_config: Divide width by 10 to 20 / Generally have 128 as width ; Adapt num_attention_heads, too (128 hidden & 8 heads)
base_config -> delta_config: Multiply hidden size by 2

Do small HP optim on LR at small scale:
Run tiny grid search at 64 hidden size (200M params) on init std 0.1 / default; make same warmup as prior experiments; Use batch size from prior experiments; Use cosine deacying to 10%

Then use those HPs found for 1B & 2b8 models
"""


### Cosine Annealing with Warmup from
### https://github.com/Lightning-Universe/lightning-bolts/blob/master/pl_bolts/optimizers/lr_scheduler.py

from torch import nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler

class LinearWarmupCosineAnnealingLR(_LRScheduler):
    """Sets the learning rate of each parameter group to follow a linear warmup schedule between warmup_start_lr
    and base_lr followed by a cosine annealing schedule between base_lr and eta_min.
    .. warning::
        It is recommended to call :func:`.step()` for :class:`LinearWarmupCosineAnnealingLR`
        after each iteration as calling it after each epoch will keep the starting lr at
        warmup_start_lr for the first epoch which is 0 in most cases.
    .. warning::
        passing epoch to :func:`.step()` is being deprecated and comes with an EPOCH_DEPRECATION_WARNING.
        It calls the :func:`_get_closed_form_lr()` method for this scheduler instead of
        :func:`get_lr()`. Though this does not change the behavior of the scheduler, when passing
        epoch param to :func:`.step()`, the user should call the :func:`.step()` function before calling
        train and validation methods.
    Example:
        >>> layer = nn.Linear(10, 1)
        >>> optimizer = Adam(layer.parameters(), lr=0.02)
        >>> scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=10, max_epochs=40)
        >>> #
        >>> # the default case
        >>> for epoch in range(40):
        ...     # train(...)
        ...     # validate(...)
        ...     scheduler.step()
        >>> #
        >>> # passing epoch param case
        >>> for epoch in range(40):
        ...     scheduler.step(epoch)
        ...     # train(...)
        ...     # validate(...)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int,
        max_epochs: int,
        warmup_start_lr: float = 0.0,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        """
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            warmup_epochs (int): Maximum number of iterations for linear warmup
            max_epochs (int): Maximum number of iterations
            warmup_start_lr (float): Learning rate to start the linear warmup. Default: 0.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
        """
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min

        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, " "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == 0:
            return [self.warmup_start_lr] * len(self.base_lrs)
        if self.last_epoch < self.warmup_epochs:
            return [
                group["lr"] + (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        if self.last_epoch == self.warmup_epochs:
            return self.base_lrs
        if (self.last_epoch - 1 - self.max_epochs) % (2 * (self.max_epochs - self.warmup_epochs)) == 0:
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / (self.max_epochs - self.warmup_epochs))) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]

        return [
            (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            / (
                1
                + math.cos(
                    math.pi * (self.last_epoch - self.warmup_epochs - 1) / (self.max_epochs - self.warmup_epochs)
                )
            )
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self) -> List[float]:
        """Called when epoch is passed as a param to the `step` function of the scheduler."""
        if self.last_epoch < self.warmup_epochs:
            return [
                self.warmup_start_lr + self.last_epoch * (base_lr - self.warmup_start_lr) / (self.warmup_epochs - 1)
                for base_lr in self.base_lrs
            ]

        return [
            self.eta_min
            + 0.5
            * (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)))
            for base_lr in self.base_lrs
        ]


from mutransformers import GPT2Config, GPT2LMHeadModel
from mup import make_base_shapes, set_base_shapes, MuAdamW
from functools import partial


TARGET_CONFIG = {
    "200M": {
        "hidden_size": 1024,
        "intermediate_size": 1024*4,
        "num_attention_heads": 32,
        "num_layers": 12,
        "batch_size": 256,
    },
    "1B1": {
        "hidden_size": 1792,
        "intermediate_size": 1792*4,
        "num_attention_heads": 14,
        "num_layers": 26,
        "batch_size": 256,
    }
    "2B8": {
        "hidden_size": 2560,
        "intermediate_size": 2560*4,
        "num_attention_heads": 20,
        "num_layers": 34,
        "batch_size": 512,
    },
}

BASE_HIDDEN = 128
BASE_INTERMEDIATE = 256
BASE_NUM_ATTENTION_HEADS = 8
LR = 1e-3 # MUP default LR
INIT_RANGE = 0.02 # MUP default init range

CONFIG_TO_RUN = "200M" # MODIFY BASED ON DESIRED CONFIG
RUN_OFFLINE = True

BATCH_SIZE = TARGET_CONFIG[CONFIG_TO_RUN]["batch_size"]


# define a base model
base_config = GPT2Config(
    hidden_size=BASE_HIDDEN,
    intermediate_size=BASE_INTERMEDIATE,
    num_attention_heads=BASE_NUM_ATTENTION_HEADS,
    initializer_range=INIT_RANGE,    
)
base_model = GPT2LMHeadModel(config=base_config)
# define a delta models where we vary all "widths" we want to vary
delta_config = GPT2Config(
    hidden_size=BASE_HIDDEN*2,
    intermediate_size=BASE_INTERMEDIATE*2,
    num_attention_heads=BASE_NUM_ATTENTION_HEADS*2,
    initializer_range=INIT_RANGE,
)
delta_model = GPT2LMHeadModel(config=delta_config)
# define a base shape object based on comparing delta_model against base_model
base_shapes = make_base_shapes(base_model, delta_model, savefile='gpt256.bsh')

# define target model
target_config = GPT2Config(
    hidden_size=TARGET_CONFIG[CONFIG_TO_RUN]["hidden_size"],
    intermediate_size=TARGET_CONFIG[CONFIG_TO_RUN]["intermediate_size"],
    num_attention_heads=TARGET_CONFIG[CONFIG_TO_RUN]["num_attention_heads"],
    num_layers=TARGET_CONFIG[CONFIG_TO_RUN]["num_layers"],
    initializer_range=INIT_RANGE,    
)
target_model = GPT2LMHeadModel(config=target_config)

# set base shapes
set_base_shapes(target_model, base_shapes)
# you can alternatively load base shape from file
# set_base_shapes(target_model, 'bert256.bsh')

# re-initialize
target_model.apply(target_model._init_weights)

# make sure to use mup optimizers for training
optimizer = MuAdamW(target_model.parameters(), lr=LR)


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
# wget https://huggingface.co/datasets/allenai/c4/resolve/main/en/c4-validation.00000-of-00008.json.gz
val_dataset = load_dataset('json', data_files='c4-validation.00000-of-00008.json.gz')['train']
# val_dataset = load_dataset('c4', 'en', split='validation[:10%]')
datasets["validation"] = val_dataset

datasets = datasets.select_columns("text")

from transformers import AutoTokenizer, Trainer, TrainingArguments, get_scheduler

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

num_steps = len(lm_datasets["train"]) // BATCH_SIZE

scheduler = LinearWarmupCosineAnnealingLR(
    optimizer,
    warmup_epochs=num_steps // 100, # 1% of training steps
    max_epochs=num_steps,
    eta_min=LR / 10, # Decay to 10% of LR
)


training_args = TrainingArguments(
    model_name,
    evaluation_strategy = "epoch",
    weight_decay=0.01,
    push_to_hub=RUN_OFFLINE,
    per_device_train_batch_size=BATCH_SIZE,
    warmup_ratio=0.01, # 10% of training steps
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=target_model,
    args=training_args,
    train_dataset=lm_datasets["train"],
    eval_dataset=lm_datasets["validation"],
    optimizers=(optimizer, scheduler), # Use mup optimizer & cosine scheduler
)

trainer.train()

import math
eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.4f}")


if RUN_OFFLINE:
    trainer.save_model(model_name)
else:
    trainer.push_to_hub()
