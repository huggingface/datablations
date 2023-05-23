#!/bin/bash
#SBATCH --account=project_462000119
#SBATCH --cpus-per-task=20
#SBATCH --partition=eap
#SBATCH --gres=gpu:mi250:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/install_apex.out
#SBATCH --error=logs/install_apex.err

module --quiet purge
module load cray-python

source venv/bin/activate

cd apex
python setup.py install --cpp_ext --cuda_ext
