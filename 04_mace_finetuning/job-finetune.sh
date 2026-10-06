#!/bin/bash
#SBATCH --qos=regular
##SBATCH --account=pintxo
##SBATCH --partition=pintxo
#SBATCH --job-name=training-mace4 # Job name
#SBATCH --gres=gpu:1 #
#SBATCH --mem=24G
#SBATCH --nodes=1 # Number of nodes to be used
#SBATCH --ntasks-per-node=1 # Total number of cores
#SBATCH --cpus-per-task=1 # Total number of cores
#SBATCH --output=finetune_%j.log # Standard output and error log
pwd; hostname; date

echo "In nodes $SLURM_NODELIST"

# set up environment
module purge
module load Miniforge3
source ~/.venv_macegpu/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK # run

mace_run_train \
    --name="finetune-MATPES-PBE-01-6" \
    --default_dtype="float64" \
    --model="MACE" \
    --foundation_model="foundation-model/MACE-matpes-pbe-omat-ft.model" \
    --log_dir="logs_finetune" \
    --multiheads_finetuning=False \
    --checkpoints_dir="checkpoints_finetune" \
    --r_max=5.0 \
    --num_interactions=3 \
    --num_channels=64 \
    --max_L=2 \
    --correlation=3 \
    --train_file="training-set/data/train.extxyz" \
    --valid_file="training-set/data/valid.extxyz" \
    --E0s='{6:-157.78736023214321, 8:-436.05186387233607, 1:-13.726268701299707}' \
    --energy_key="energy" \
    --forces_key="forces" \
    --energy_weight=1.0 \
    --forces_weight=10.0 \
    --lr=0.0003 \
    --batch_size=4 \
    --max_num_epochs=100 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --device=cuda \
    --seed=1001123



echo DONE
date
