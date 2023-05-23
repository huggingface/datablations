#!/bin/bash

# Launch script using torch.distributed.run(). Used by slurm
# scripts, don't invoke directly.

# Samuel's fix for apparent error in SLURM initialization
if [ $SLURM_LOCALID -eq 0 ]; then
    rm -rf /dev/shm/*
    rocm-smi || true
else
    sleep 2
fi

export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export FI_CXI_DEFAULT_CQ_SIZE=131072

# debugging (noisy)
#export NCCL_DEBUG=INFO
#export RCCL_KERNEL_COLL_TRACE_ENABLE=1
#export NCCL_DEBUG_SUBSYS=INIT,COLL

module --quiet purge
module load cray-python

module load CrayEnv
module load PrgEnv-cray/8.3.3
module load craype-accel-amd-gfx90a
module load cray-python

module load rocm/5.2.3
module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
module load aws-ofi-rccl/rocm-5.2.3

source venv/bin/activate

MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=9999

echo "Launching on $SLURMD_NODENAME ($SLURM_PROCID/$SLURM_JOB_NUM_NODES)," \
     "master $MASTER_NODE port $MASTER_PORT," \
     "GPUs $SLURM_GPUS_ON_NODE," \
     "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -u -m torch.distributed.run \
    --nnodes $SLURM_JOB_NUM_NODES \
    --nproc_per_node $SLURM_GPUS_ON_NODE \
    --node_rank=$SLURM_PROCID \
    --master_addr $MASTER_NODE \
    --master_port $MASTER_PORT \
    "$@"