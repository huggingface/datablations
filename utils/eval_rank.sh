#!/bin/bash
#SBATCH --exclude=nid005159
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH -p small-g
#SBATCH -t 2-0:00:00
#SBATCH --gpus-per-node=mi250:0
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000119
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# if run without sbatch, invoke here
if [ -z $SLURM_JOB_ID ]; then
    mkdir -p logs
    sbatch "$0"
    exit
fi

set -euo pipefail

# symlink logs/latest_eval.out and logs/latest_eval.err
ln -f -s $SLURM_JOB_ID.out logs/latest_eval.out
ln -f -s $SLURM_JOB_ID.err logs/latest_eval.err

source /pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/venv/bin/activate

echo "START TIME: $(date)"

# defining the right environment variables
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/scratch/project_462000119/ds_cache

CKPTS=(
/pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-4b2-84b-oscarseeds/4b284b21boscarseed3/global_step80108
/pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-4b2-84b-oscarseeds/4b284b21boscarseed2/global_step80108
/pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-4b2-84b-oscarseeds/4b284b21boscarseed4/global_step80108
/pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-4b2-84b-oscarseeds/4b284b28boscarseed1/global_step80108
)

FEWSHOT_CONFIGS=(
0
1
2
3
4
5
)

# Iterate through all possible combinations of data config, model ckpt & fewshot config and run the jobs
for ((i=0; i<${#CKPTS[@]}; i++)); do
    for ((j=0; j<${#FEWSHOT_CONFIGS[@]}; j++)); do
        #echo "sbatch --export=CKPT=${CKPTS[$i]},FEWSHOT_CONFIG=${FEWSHOT_CONFIGS[$j]},DATASET=${DATASETS[$k]} eval.sh"
        MODEL_CKPT=${CKPTS[$i]}
        MODEL_CKPT_NO_STEP=${MODEL_CKPT%/*}
        MODEL_NAME=${MODEL_CKPT_NO_STEP##*/}
        mkdir -p $MODEL_CKPT_NO_STEP/evaluation/rankeval
        #mv $MODEL_CKPT_NO_STEP/evaluation/$MODEL_NAME\_${FEWSHOT_CONFIGS[$j]}.* $MODEL_CKPT_NO_STEP/evaluation/rankeval/
        OUTPUT_PATH=$MODEL_CKPT_NO_STEP/evaluation/rankeval/$MODEL_NAME\_${FEWSHOT_CONFIGS[$j]}.json
        eval_script="./eval_$i-$j.slurm"
        cat <<EOT > $eval_script
#!/bin/bash
#SBATCH --exclude=nid005159
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH -p small-g
#SBATCH -t 2-0:00:00
#SBATCH --gpus-per-node=mi250:1
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --account=project_462000119
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/scratch/project_462000119/ds_cache

VOCAB_FILE="gpt2/vocab.json"
MERGE_FILE="gpt2/merges.txt"

PP_SIZE=1
TP_SIZE=1
# different from the training MICRO_BATCH_SIZE - no optim memory, so can do bigger BS
# make as big as it can fit into gpu w/o OOM, but not too close to 100%
EVAL_MICRO_BATCH_SIZE=1
MICRO_BS_MULTIPLIER=1

# Model parameters
SEQ_LEN=2048

# Dummy arguments
MEGATRON_REQUIRED_ARGS=" \
    --num-layers -1 \
    --hidden-size -1 \
    --num-attention-heads -1 \
    --seq-length -1  \
    --max-position-embeddings -1 \
"

ZERO_STAGE=0

mkdir -p ds_configs
DS_CONFIG_PATH="ds_configs/\$SLURM_JOB_ID.json"

cat <<EOF > "\$DS_CONFIG_PATH"
{
    "train_micro_batch_size_per_gpu": 1,
    "train_batch_size": 1,
    "gradient_clipping": 1.0,
    "zero_optimization": {
        "stage": \$ZERO_STAGE
    },
    "bf16": {
        "enabled": true
    },
    "steps_per_print": 2000,
    "wall_clock_breakdown": false
}
EOF

DEEPSPEED_ARGS=" \
    --deepspeed \
    --deepspeed_config \$DS_CONFIG_PATH \
    --zero-stage \$ZERO_STAGE \
    "

CMD="Megatron-DeepSpeed/tasks/eval_harness/evaluate.py \
    --load $MODEL_CKPT \
    --results_path $OUTPUT_PATH \
    --tensor-model-parallel-size \$TP_SIZE  \
    --pipeline-model-parallel-size \$PP_SIZE \
    --vocab-file \$VOCAB_FILE \
    --merge-file \$MERGE_FILE \
    --micro-batch-size \$EVAL_MICRO_BATCH_SIZE \
    --no-load-optim \
    --no-load-rng \
    --bf16 \
    --inference \
    --seq-length \$SEQ_LEN \
    --task_list anli_r1,anli_r2,anli_r3,cb,copa,hellaswag,rte,winogrande,storycloze_2016,boolq,arc_easy,arc_challenge,sciq,piqa \
    --intermed_results \
    --adaptive_seq_len \
    --micro_bs_multiplier \$MICRO_BS_MULTIPLIER \
    --fewshots ${FEWSHOT_CONFIGS[$j]} \
    \$MEGATRON_REQUIRED_ARGS \
    \$DEEPSPEED_ARGS \
    "

echo "\$CMD"

echo "START \$SLURM_JOBID: $(date)"

srun --label launch.sh \$CMD

echo "END \$SLURM_JOBID: $(date)"
EOT
        sbatch $eval_script
        # Sleep for a bit to avoid hitting the job submission limit
        sleep 0.1
    done
done
