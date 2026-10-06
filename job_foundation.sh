#!/bin/bash
#SBATCH --qos=regular
##SBATCH --partition=pintxo
##SBATCH --account=pintxo
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --ntasks-per-node=1         
#SBATCH --nodes=1         
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=16:10:00          # total run time limit (HH:MM:SS)
#SBATCH --job-name="10-1N-MD-mace"
#SBATCH --gres=gpu:1        # number of gpus per node
pwd; hostname; date

module load Miniforge3

source ~/.venv/bin/activate

python md_foundation.py

date
