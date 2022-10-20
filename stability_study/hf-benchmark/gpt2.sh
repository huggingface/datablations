# export CUDA_VISIBLE_DEVICES=0 
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export TRANSFORMERS_CACHE=/scratch/project_462000119/tf_cache
export TRANSFORMERS_CACHE=/scratch/project_462000119/tf_cache/models
export HF_DATASETS_CACHE=/scratch/project_462000119/tf_cache/datasets
export HF_MODULES_CACHE=/scratch/project_462000119/tf_cache/modules
export HF_METRICS_CACHE=/scratch/project_462000119/tf_cache/metrics

python \
transformers/scripts/benchmark/trainer-benchmark.py \
--base-cmd \
' \
transformers/examples/pytorch/language-modeling/run_clm.py --model_type gpt2 --tokenizer_name gpt2 --dataset_name stas/openwebtext-10k --logging_strategy steps --logging_steps 1 --log_level debug --report_to tensorboard --save_strategy no --do_train --max_train_samples 1000000 --per_device_train_batch_size 8 --num_train_epochs 10 --warmup_steps 8 --block_size 512 \
' \
--target-metric-key train_samples_per_second --repeat-times 1 --variations \
'--optim adamw_torch' \
--report-metric-keys train_loss --base-variation '--optim adamw_torch' \
--verbose \
--output_dir hf_trainer_benchmark/gpt2