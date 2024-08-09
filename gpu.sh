#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 2:00:00
#SBATCH --gpus=v100-16:1

#type 'man sbatch' for more information and options
#this job will ask for 1 full v100-32 GPU node(8 V100 GPUs) for 5 hours
#this job would potentially charge 40 GPU SUs

# make outfile and errfile:
#SBATCH -o outfiles/gpu%j.out
#SBATCH -e outfiles/gpu%j.err

#echo commands to stdout
set -x
conda init
conda activate iaifi
# move to working directory
# this job assumes:
# - all input data is stored in this directory
# - all output should be stored in this directory
# - please note that groupname should be replaced by your groupname
# - username should be replaced by your username
# - path-to-directory should be replaced by the path to your directory where the executable is


python wandb_sweep.py &
python wandb_sweep.py &
python wandb_sweep.py
