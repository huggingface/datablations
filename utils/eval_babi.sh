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

CMDS=(
"python main.py --no_cache --model gpt2 --model_args pretrained=/pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-2b8-55b-oscarroots/transformers --tasks babi --limit 3000 --num_fewshot 0 --output_path /pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-2b8-55b-oscarroots/evaluation/lm1-2b8-55b-oscarroots_0_babi.json"
"python main.py --no_cache --model gpt2 --model_args pretrained=/pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-2b8-55b-oscarroots/transformers --tasks babi --limit 3000 --num_fewshot 1 --output_path /pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-2b8-55b-oscarroots/evaluation/lm1-2b8-55b-oscarroots_1_babi.json"
"python main.py --no_cache --model gpt2 --model_args pretrained=/pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-2b8-55b-oscarroots/transformers --tasks babi --limit 3000 --num_fewshot 2 --output_path /pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-2b8-55b-oscarroots/evaluation/lm1-2b8-55b-oscarroots_2_babi.json"
"python main.py --no_cache --model gpt2 --model_args pretrained=/pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-2b8-55b-oscarroots/transformers --tasks babi --limit 3000 --num_fewshot 3 --output_path /pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-2b8-55b-oscarroots/evaluation/lm1-2b8-55b-oscarroots_3_babi.json"
"python main.py --no_cache --model gpt2 --model_args pretrained=/pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-2b8-55b-oscarroots/transformers --tasks babi --limit 3000 --num_fewshot 4 --output_path /pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-2b8-55b-oscarroots/evaluation/lm1-2b8-55b-oscarroots_4_babi.json"
"python main.py --no_cache --model gpt2 --model_args pretrained=/pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-2b8-55b-oscarroots/transformers --tasks babi --limit 3000 --num_fewshot 5 --output_path /pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lm1-2b8-55b-oscarroots/evaluation/lm1-2b8-55b-oscarroots_5_babi.json"
)

# Iterate through all possible combinations of data config, model ckpt & fewshot config and run the jobs
for ((i=0; i<${#CMDS[@]}; i++)); do
    eval_script="./eval_$i.slurm"
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
set -euo pipefail
# symlink logs/latest_eval.out and logs/latest_eval.err
ln -f -s "\$SLURM_JOB_ID.out" logs/latest_eval.out
ln -f -s "\$SLURM_JOB_ID.err" logs/latest_eval.err
source /pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/venv/bin/activate
echo "START TIME: $(date)"
# defining the right environment variables
export HF_DATASETS_OFFLINE=1
export HF_DATASETS_CACHE=/scratch/project_462000119/ds_cache
# Converted transformer checkpoint
cd /pfs/lustrep4/scratch/project_462000119/muennighoff/nov-2022-bettercom/lmevalbabi/lm-evaluation-harness

${CMDS[$i]}

echo "END TIME: $(date)"
EOT
    # Submit the job
    sbatch $eval_script
    # Sleep for a bit to avoid hitting the job submission limit
    sleep 0.1
done
