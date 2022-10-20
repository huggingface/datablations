#!/bin/bash

#SBATCH --job-name=ds_pretrain_gpt2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=256G
#SBATCH -p pilot
#SBATCH -t 12:00:00
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
export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions/
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

CONFIG=baseline
TAG=baseline
MODEL_SIZE=345
LR=1.5e-4
TOTAL_BATCHSIZE=512
SEQ_LEN=1024
MP_SIZE=1
SEED=1234
SAVE_INTERVAL=5000
NUM_ITER=600000
NUM_TOKEN=157286400000
LR_DECAY_TOKEN=157286400000
LR_WARMUP_ITER=3000
CONFIG_TEMPLATE=false
CURRICULUM_STEP=0
CURRICULUM_MIN=0

# 12-layer, 768-hidden, 12-heads, 117M parameters
# 24-layer, 1024-hidden, 16-heads, 345M parameters
# 36-layer, 1280-hidden, 20-heads, 774M parameters
# 48-layer, 1600-hidden, 25-heads, 1558M parameters
if [[ $MODEL_SIZE -eq 117 ]]; then
        NUM_LAYERS=12
        HIDDEN_SIZE=768
        NUM_ATTN_HEADS=12
elif [[ $MODEL_SIZE -eq 345 ]]; then
        NUM_LAYERS=24
        HIDDEN_SIZE=1024
        NUM_ATTN_HEADS=16
elif [[ $MODEL_SIZE -eq 774 ]]; then
        NUM_LAYERS=36
        HIDDEN_SIZE=1280
        NUM_ATTN_HEADS=20
elif [[ $MODEL_SIZE -eq 1558 ]]; then
        NUM_LAYERS=48
        HIDDEN_SIZE=1600
        NUM_ATTN_HEADS=25
else
        echo "Model size not supported."
        exit 1
fi

# Pipeline parallelism. 1 means no pipelines.
PP_SIZE=1

# Change for multinode config
NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
NUM_GPUS=$(( ${NUM_WORKERS} * ${NUM_GPUS_PER_WORKER} ))
if [[ $PP_SIZE -gt 0 ]]; then
    DP_SIZE=$(( ${NUM_GPUS} / (${PP_SIZE} * ${MP_SIZE}) ))
else
    DP_SIZE=$(( ${NUM_GPUS} / ${MP_SIZE} ))
fi
# Batch size per gpu, here we assume grad accumulation step 1
# you can reduce this if gpu OOM
BATCHSIZE=$((TOTAL_BATCHSIZE/DP_SIZE))

VOCAB_FILE="/scratch/project_462000119/sampo/gpt-hf-settings-no-tp-pp-3/gpt2-vocab.json"
MERGE_FILE="/scratch/project_462000119/sampo/gpt-hf-settings-no-tp-pp-3/gpt2-merges.txt"
DATA_PATH="/project/project_462000119/nouatazi/datablations/stability_study/data/owt2"


#ZeRO Configs
stage=1

current_time=$(date "+%Y.%m.%d-%H.%M.%S")
script_path=$(realpath $0)
script_dir=$(dirname $script_path)
host="${HOSTNAME}"

if [ "${CONFIG_TEMPLATE}" = "true" ]; then
template_json="$script_dir/ds_zero_stage_${stage}_config_${CONFIG}.json"
config_json="$script_dir/ds_zero_stage_${stage}_config_${CONFIG}_min${CURRICULUM_MIN}_max${SEQ_LEN}_step${CURRICULUM_STEP}.json"
sed "s/CONFIG_CL_MIN/${CURRICULUM_MIN}/" ${template_json} \
    | sed "s/CONFIG_CL_MAX/${SEQ_LEN}/" \
    | sed "s/CONFIG_CL_DURATION/${CURRICULUM_STEP}/" \
	  > ${config_json}
else
config_json="/project/project_462000119/nouatazi/datablations/stability_study/meg-lm/ds_zero_stage_${stage}_config_${CONFIG}.json"
fi

JOB_NAME="gpt2_${MODEL_SIZE}M_bsz${TOTAL_BATCHSIZE}_seq${SEQ_LEN}_lr${LR}_warmup${LR_WARMUP_ITER}_decay${LR_DECAY_TOKEN}_seed${SEED}_${TAG}_stage${stage}_n${NUM_WORKERS}_g${NUM_GPUS_PER_WORKER}_mp${MP_SIZE}"
LOG_NAME="${JOB_NAME}_${host}_${current_time}"

OUTPUT_BASEPATH="ds_test/${JOB_NAME}"
mkdir -p "${OUTPUT_BASEPATH}/tensorboard/curriculum/"
mkdir -p "${OUTPUT_BASEPATH}/checkpoint/curriculum/"
mkdir -p "${OUTPUT_BASEPATH}/log/curriculum/"
LOGDIR="${OUTPUT_BASEPATH}/tensorboard/curriculum/${LOG_NAME}"
CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint/curriculum/${JOB_NAME}"

gpt_options=" \
        --tensor-model-parallel-size ${MP_SIZE} \
        --num-layers $NUM_LAYERS \
        --hidden-size $HIDDEN_SIZE \
        --num-attention-heads $NUM_ATTN_HEADS \
        --seq-length $SEQ_LEN \
        --max-position-embeddings $SEQ_LEN \
        --micro-batch-size $BATCHSIZE \
        --global-batch-size ${TOTAL_BATCHSIZE} \
        --train-iters $NUM_ITER \
        --train-tokens $NUM_TOKEN \
        --lr-decay-tokens $LR_DECAY_TOKEN \
        --save $CHECKPOINT_PATH \
        --data-path $DATA_PATH \
        --vocab-file $VOCAB_FILE \
        --merge-file $MERGE_FILE \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --override-lr-scheduler \
        --lr $LR \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --lr-warmup-iters $LR_WARMUP_ITER \
        --checkpoint-activations \
        --log-interval 100 \
        --save-interval $SAVE_INTERVAL \
        --eval-interval 100 \
        --eval-iters 10 \
        --fp16 \
        --seed $SEED \
        --tensorboard-queue-size 1 \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --no-masked-softmax-fusion \
        --tensorboard-dir ${LOGDIR}
"

deepspeed_options=" \
        --deepspeed \
        --deepspeed_config ${config_json} \
        --zero-stage ${stage} \
        --pipeline-model-parallel-size ${PP_SIZE} \
        --deepspeed-activation-checkpointing
"

full_options="${gpt_options} ${deepspeed_options}"

run_cmd="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER}  Megatron-LM/pretrain_gpt.py ${full_options}"
echo ${run_cmd}
eval ${run_cmd}

set +x
