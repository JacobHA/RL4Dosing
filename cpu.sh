#!/bin/bash
#SBATCH --job-name=eval-finetuned
#SBATCH --time=23:00:00
#SBATCH --mem-per-cpu=2gb
#SBATCH --cpus-per-task=4

# Set filenames for stdout and stderr.  %j can be used for the jobid.
# see "filename patterns" section of the sbatch man page for
# additional options
#SBATCH --error=outfiles/%j.err
#SBATCH --output=outfiles/%j.out
##SBATCH --partition=AMD6276
##SBATCH --partition=Intel
##SBATCH --partition=AMD6128

#SBATCH --array=1-20
## --begin=now+1min
echo "using scavenger"

# Prepare conda:
eval "$(conda shell.bash hook)"
conda activate /home/jacob.adamczyk001/miniconda3/envs/oblenv
export CPATH=$CPATH:$CONDA_PREFIX/include
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export MUJOCO_GL="glfw"


echo "Start Run"
echo `date`

python wandb_sweep.py