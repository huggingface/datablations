#!/bin/bash

# Set up virtual environment for Megatron-DeepSpeed pretrain_gpt.py.

# This script creates the directories venv and apex. If either of
# these exists, ask to delete.
for p in venv apex; do
    if [ -e "$p" ]; then
	read -n 1 -r -p "$p exists. OK to remove? [y/n] "
	echo
	if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Removing $p."
	    rm -rf "$p"
	else
            echo "Exiting."
            exit 1
	fi
    fi
done

# Load modules
module --quiet purge

module load CrayEnv
module load PrgEnv-cray/8.3.3
module load craype-accel-amd-gfx90a
module load cray-python

module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
module load suse-repo-deps/sam-default
module load rocm/sam-5.2.3.lua
module load rccl/sam-develop.lua
module load aws-ofi-rccl/sam-default.lua

# Create and activate venv
python -m venv --system-site-packages venv
source venv/bin/activate

# Upgrade pip etc.
python -m pip install --upgrade pip setuptools wheel

# Install pip packages
python -m pip install --upgrade torch --extra-index-url https://download.pytorch.org/whl/rocm5.2/
python -m pip install --upgrade numpy datasets evaluate accelerate sklearn nltk
python -m pip install --upgrade git+https://github.com/huggingface/transformers
python -m pip install --upgrade deepspeed

# Install apex on a GPU node
git clone https://github.com/ROCmSoftwarePlatform/apex/

# Use specific working commit
cd apex
git checkout 5de49cc90051adf094920675e1e21175de7bad1b
cd -

mkdir -p logs
cat <<EOF > install_apex.sh
#!/bin/bash
#SBATCH --account=project_462000119
#SBATCH --cpus-per-task=20
#SBATCH --partition=eap
#SBATCH --gres=gpu:mi250:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/install_apex.out
#SBATCH --error=logs/install_apex.err

module --quiet purge

module load CrayEnv
module load PrgEnv-cray/8.3.3
module load craype-accel-amd-gfx90a
module load cray-python

module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
module load suse-repo-deps/sam-default
module load rocm/sam-5.2.3.lua
module load rccl/sam-develop.lua
module load aws-ofi-rccl/sam-default.lua

source venv/bin/activate

cd apex
python setup.py install --cpp_ext --cuda_ext
EOF

echo "Installing apex on a GPU node. This is likely to take around 30 min."
time sbatch --wait install_apex.sh
