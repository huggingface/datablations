# Fast Setup instructions

(These instructions are taken from https://github.com/TurkuNLP/finngen-tools/blob/main/lumi_start_fast.md, originally adapted from https://github.com/bigscience-workshop/Megatron-DeepSpeed/blob/main/start_fast.md for LUMI)

This quick instructions document contains 3 steps:

1. installing software
2. preparing data
3. running the script

## 1. Software

Please follow this exact order.

1. Set up modules

```
module --quiet purge
module load cray-python
```

2. Create and activate virtual environment

```
python -m venv --system-site-packages venv
source venv/bin/activate
```

3. Upgrade and install pip packages

```
python -m pip install --upgrade pip setuptools wheel
python -m pip install --upgrade torch --extra-index-url https://download.pytorch.org/whl/rocm5.1.1
python -m pip install --upgrade numpy datasets evaluate accelerate sklearn nltk
python -m pip install --upgrade git+https://github.com/huggingface/transformers
python -m pip install --upgrade deepspeed
```

4. Install apex

This step is expected to take ~30 minutes.

(This is launched with `sbatch` because it needs to be run on a GPU node. Additionally, a specific commit is checked out because the latest commit did not compile earlier.)

```
git clone https://github.com/ROCmSoftwarePlatform/apex/

cd apex
git checkout 5de49cc90051adf094920675e1e21175de7bad1b
cd -

mkdir -p logs

cat <<EOF > install_apex.sh
#!/bin/bash
#SBATCH --account=project_462000119
#SBATCH --cpus-per-task=20
#SBATCH --partition=pilot
#SBATCH --gres=gpu:mi250:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/install_apex.out
#SBATCH --error=logs/install_apex.err

module --quiet purge
module load cray-python

source venv/bin/activate

cd apex
python setup.py install --cpp_ext --cuda_ext
EOF

time sbatch --wait install_apex.sh
```

5. Checkout `Megatron-DeepSpeed` and install its requirements

Note: we won't `pip install -r requirements.txt` because we need a ROCm version of torch for LUMI.

```
git clone https://github.com/bigscience-workshop/Megatron-DeepSpeed
python -m pip install parameterized regex six tensorboard
```

## 2. Data

Will work under the `Megatron-DeepSpeed` clone

```
cd Megatron-DeepSpeed
```

Prepare data for preprocessing

```
mkdir -p data
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json -O data/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt -O data/gpt2-merges.txt
python -c 'from datasets import load_dataset; ds = load_dataset("stas/oscar-en-10k", split="train", keep_in_memory=False); ds.to_json(f"data/oscar-en-10k.jsonl", orient="records", lines=True, force_ascii=False)'
```

Pre-process a small dataset to be used for training

```
python tools/preprocess_data.py \
    --input data/oscar-en-10k.jsonl \
    --output-prefix data/meg-gpt2-oscar-en-10k \
    --dataset-impl mmap \
    --tokenizer-type GPT2BPETokenizer \
    --merge-file data/gpt2-merges.txt \
    --vocab data/gpt2-vocab.json \
    --append-eod \
    --workers 4
```

now you have data/meg-gpt2-oscar-en-10k, vocab and merges files to pass as arguments to training, the next section shows how to use them.

Note that Megatron wants `data/meg-gpt2-oscar-en-10k_text_document` prefix later in `--data-path`

## 3. Train

Work in the directory with the venv instead of the `Megatron-DeepSpeed` subdirectory

```
cd ..
```

Here is a tiny model training setup configured over 2 gpus to train on the data we prepared in step 2.

Save this as a script and execute with `sbatch`, e.g. save as `pretrain_gpt.sh` and run `sbatch pretrain_gpt.sh`.

```
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH -p pilot
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=mi250:2
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000119
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

module --quiet purge
module load cray-python

source venv/bin/activate

set -euo pipefail

CHECKPOINT_PATH=Megatron-DeepSpeed/checkpoints/gpt2

VOCAB_FILE=Megatron-DeepSpeed/data/gpt2-vocab.json
MERGE_FILE=Megatron-DeepSpeed/data/gpt2-merges.txt
DATA_PATH=Megatron-DeepSpeed/data/meg-gpt2-oscar-en-10k_text_document
TENSORBOARD_PATH=Megatron-DeepSpeed/output_dir/tensorboard

N_GPUS=2
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=16
TP_SIZE=2
PP_SIZE=1

NLAYERS=2
NHIDDEN=8
NHEADS=2
SEQ_LEN=512
VOCAB_SIZE=50257

SAVE_INTERVAL=50

TRAIN_SAMPLES=10_000

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --rampup-batch-size 2 2 1_000 \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-samples $TRAIN_SAMPLES \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-8 \
    --lr 1e-4 \
    --lr-warmup-samples 5 \
    --min-lr 1e-6 \
    --lr-decay-style cosine \
    --lr-decay-samples 12 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    --embed-layernorm \
    --fp16 \
    --partition-activations \
    --seed 42 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    "

OUTPUT_ARGS=" \
    --exit-interval 100 \
    --log-interval 10 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 100 \
    --eval-iters 10 \
    --checkpoint-activations \
    "

DATA_ARGS=" \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    --kill-switch-path /tmp/kill-switch \
    "

ZERO_STAGE=1

config_json="./ds_config.json"

# Deepspeed figures out GAS dynamically from dynamic GBS via set_train_batch_size()
cat <<EOT > $config_json
{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": $ZERO_STAGE
  },
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  },
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}
EOT

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config ${config_json} \
    --zero-stage ${ZERO_STAGE} \
    --deepspeed-activation-checkpointing \
    "

ALL_ARGS="$GPT_ARGS $OUTPUT_ARGS $DATA_ARGS $DEEPSPEED_ARGS"

MASTER_ADDR=localhost
MASTER_PORT=6777

export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node $N_GPUS \
    --nnodes 1 \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    --rdzv_backend c10d \
    --max_restarts 0 \
    --tee 3 \
    "
export CMD=" \
    $LAUNCHER Megatron-DeepSpeed/pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    --distributed-backend nccl \
    $ALL_ARGS \
    "

echo $CMD

$CMD
```
