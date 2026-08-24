#!/bin/bash
#SBATCH --time 00:10:00
#SBATCH --nodes 1
#SBATCH --partition maxgpu
#SBATCH --job-name time_diffusion
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=henry.day-hall@desy.de
#SBATCH --output /data/dust/user/dayhallh/data/CaloClouds_diffusion/joblogs/%j_%a.out
#SBATCH --error /data/dust/user/dayhallh/data/CaloClouds_diffusion/joblogs/%j_%a.err
#SBATCH --constraint="GPUx1&A100-PCIE-80GB"
#SBATCH --array=0

module load maxwell mamba
. mamba-init
cd ~/training/point-cloud-diffusion
mamba activate caloi

cd /home/dayhallh/training/CC_ExpSpec/CaloCloudi

# if SLURM_ARRAY_TASK_ID is not set, it will be 0
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=0
fi

# Check if task ID is within range
if [ "$SLURM_ARRAY_TASK_ID" -ge "$total_jobs" ]; then
    echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID out of range (0-$((total_jobs-1)))" >&2
    exit 0
fi

config="/data/dust/user/dayhallh/data/CaloClouds_diffusion/logs/2026_08_21__10_32_03/config.yaml"
version="/home/dayhallh/training/CC_ExpSpec/CaloCloudi/distilled.ts.pt"

# give the start and end times of the whole thing
time_start=$SECONDS
time_allocated=$((60*45))
python3 scripts/generate_timings.py $time_allocated $config $version
time_end=$SECONDS
echo Ran for $((time_end-time_start))
