import os
import subprocess

CHECKPOINT_PATH="Megatron-DeepSpeed/checkpoints/gpt2"

VOCAB_FILE="/project/project_462000119/nouatazi/data/gpt2/vocab.json"
MERGE_FILE="/project/project_462000119/nouatazi/data/gpt2/merges.txt"
DATA_PATH="/scratch/project_462000119/data/pile/megatron_data/meg-gpt2_pile_text_document"
TENSORBOARD_PATH="Megatron-DeepSpeed/output_dir/tensorboard"

def makejob(N_GPUS=2,
            CHECKPOINT_PATH=CHECKPOINT_PATH, 
            VOCAB_FILE=VOCAB_FILE, 
            MERGE_FILE=MERGE_FILE, 
            DATA_PATH=DATA_PATH, 
            TENSORBOARD_PATH=TENSORBOARD_PATH,
            MICRO_BATCH_SIZE=1,
            GLOBAL_BATCH_SIZE=16,
            TP_SIZE=2,
            PP_SIZE=1,
            NLAYERS=2,
            NHIDDEN=8,
            NHEADS=2,
            SEQ_LEN=512,
            VOCAB_SIZE=50257,
            SAVE_INTERVAL=50,
            TRAIN_SAMPLES="10_000"
            ):
    return f"""#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=400G
#SBATCH -p pilot
#SBATCH -t 12:00:00
#SBATCH --gpus-per-node=mi250:{N_GPUS}
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000119
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

module --quiet purge
module load cray-python

source {os.environ['VIRTUAL_ENV']}/bin/activate

set -euo pipefail

CHECKPOINT_PATH={CHECKPOINT_PATH}

VOCAB_FILE={VOCAB_FILE}
MERGE_FILE={MERGE_FILE}
DATA_PATH={DATA_PATH}
TENSORBOARD_PATH={TENSORBOARD_PATH}

N_GPUS={N_GPUS}
MICRO_BATCH_SIZE={MICRO_BATCH_SIZE}
GLOBAL_BATCH_SIZE={GLOBAL_BATCH_SIZE}
TP_SIZE={TP_SIZE}
PP_SIZE={PP_SIZE}

NLAYERS={NLAYERS}
NHIDDEN={NHIDDEN}
NHEADS={NHEADS}
SEQ_LEN={SEQ_LEN}
VOCAB_SIZE={VOCAB_SIZE}

SAVE_INTERVAL={SAVE_INTERVAL}

TRAIN_SAMPLES={TRAIN_SAMPLES}

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
{{
  "train_micro_batch_size_per_gpu": $MICRO_BATCH_SIZE,
  "train_batch_size": $GLOBAL_BATCH_SIZE,
  "gradient_clipping": 1.0,
  "zero_optimization": {{
    "stage": $ZERO_STAGE
  }},
  "fp16": {{
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 500,
    "hysteresis": 2,
    "min_loss_scale": 1,
    "initial_scale_power": 12
  }},
  "steps_per_print": 2000,
  "wall_clock_breakdown": false
}}
EOT

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config $config_json \
    --zero-stage $ZERO_STAGE \
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

"""

def submit_job(job):
    with open('job.sbatch', 'w') as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")
    # you can check the file job.sbatch to see what is being submitted

# Ensure the log directory exists
os.system("mkdir -p logs")

# Launch the batch jobs
submit_job(makejob())

# View logs
# tail -f  logs/<JOB_ID>.out logs/<JOB_ID>.err
