#!/bin/bash

#SBATCH --job-name=blowup-3-adam
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH -p pilot
#SBATCH -t 4:00:00
#SBATCH --gpus-per-node=mi250:8
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000119
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

# if run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0"
    exit
fi

set -euo pipefail

CHECKPOINT_PATH=checkpoints
TENSORBOARD_PATH=/project/project_462000119/nouatazi/lumi_logs/tensorboard/$SLURM_JOB_NAME/$SLURM_JOB_ID
# Start from scratch
rm -rf "$CHECKPOINT_PATH"
# rm -rf "$TENSORBOARD_PATH"

# Dataj
VOCAB_FILE="/scratch/project_462000119/sampo/gpt-hf-settings-no-tp-pp-3/gpt2-vocab.json"
MERGE_FILE="/scratch/project_462000119/sampo/gpt-hf-settings-no-tp-pp-3/gpt2-merges.txt"
DATA_PATH="/project/project_462000119/nouatazi/datablations/stability_study/data/owt2"

PP_SIZE=1
TP_SIZE=1

MICRO_BATCH_SIZE=16
GLOBAL_BATCH_SIZE=512
TRAIN_ITER=73_242_187

NLAYERS=24
NHIDDEN=1024
NHEADS=16
FFN_HIDDEN_SIZE=4096
SEQ_LEN=2048

SAVE_INTERVAL=1500

OPTIMIZER_ARGS=" \
    --optimizer adam \
    --adam-beta1 0.9 \
    --adam-beta2 0.999 \
    --adam-eps 1e-8 \
    --lr 2e-4 \
    --min-lr 1e-5 \
    --lr-decay-style cosine \
    --lr-decay-samples 73_242_187 \
    --lr-warmup-samples 183_105 \
    --clip-grad 1.0 \
    --weight-decay 1e-1 \
    "

GPT_ARGS=" \
    --num-layers $NLAYERS \
    --hidden-size $NHIDDEN \
    --num-attention-heads $NHEADS \
    --ffn-hidden-size $FFN_HIDDEN_SIZE \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --rampup-batch-size 128 128 2_000_000 \
    --train-samples $TRAIN_ITER \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --loss-scale 0 \
    --clip-grad 1.0 \
    --fp16 \
    --checkpoint-activations \
    $OPTIMIZER_ARGS \
    "

OUTPUT_ARGS=" \
    --log-interval 200 \
    --save-interval $SAVE_INTERVAL \
    --eval-interval 100 \
    --eval-iters 100 \
    --tensorboard-dir $TENSORBOARD_PATH \
    --tensorboard-queue-size 5 \
    --log-timers-to-tensorboard \
    --log-batch-size-to-tensorboard \
    --log-validation-ppl-to-tensorboard \
    "

ZERO_STAGE=0

mkdir -p ds_configs
DS_CONFIG_PATH="ds_configs/$SLURM_JOB_ID.json"

cat <<EOF > $DS_CONFIG_PATH
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
EOF

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config $DS_CONFIG_PATH \
    --zero-stage $ZERO_STAGE \
    --deepspeed-activation-checkpointing \
    "

CMD=" \
    Megatron-DeepSpeed/pretrain_gpt.py \
    --tensor-model-parallel-size $TP_SIZE \
    --pipeline-model-parallel-size $PP_SIZE \
    $GPT_ARGS \
    $OUTPUT_ARGS \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH \
    --data-path $DATA_PATH \
    --data-impl mmap \
    --split 949,50,1 \
     $DEEPSPEED_ARGS \
    "

echo $CMD

echo "START $SLURM_JOBID: $(date)"

srun --label launch.sh $CMD

echo "END $SLURM_JOBID: $(date)"