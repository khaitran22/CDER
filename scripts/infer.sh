#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=test-cder
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=50G
#SBATCH -o /scratch/itee/uqptran9/Code/Work-1/Main/CDER/slurm.output
#SBATCH -e /scratch/itee/uqptran9/Code/Work-1/Main/CDER/slurm.error
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla-smx2:1
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=phankhai.tran@uq.edu.au
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

module load anaconda/3.7
source activate /scratch/itee/uqptran9/conda_env/torch/
module load cuda/10.2.89.440
module load gnu/5.4.0
module load mvapich2

srun python /scratch/itee/uqptran9/Code/Work-1/Main/CDER/train.py --data_dir ./dataset/docred/ \
 --train_file train_annotated.json \
 --dev_file dev.json \
 --test_file test.json \
 --transformer_type bert \
 --model_name_or_path bert-base-cased \
 --train_batch_size 4 \
 --test_batch_size 4 \
 --gradient_accumulation_steps 1 \
 --learning_rate 5e-5 \
 --grouped_learning_rate 1e-4 \
 --num_train_epochs 30.0 \
 --warmup_ratio 0.06 \
 --max_grad_norm 1.0 \
 --load_path ./checkpoints/cder.pt \
 --do_infer