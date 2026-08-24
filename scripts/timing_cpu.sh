#!/bin/bash
#SBATCH --time 02:45:00
#SBATCH --nodes 1
#SBATCH --partition maxcpu
#SBATCH --job-name time_diffusion
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=henry.day-hall@desy.de
#SBATCH --output /data/dust/user/dayhallh/data/CaloClouds_diffusion/joblogs/%j_%a.out
#SBATCH --error /data/dust/user/dayhallh/data/CaloClouds_diffusion/joblogs/%j_%a.err
#SBATCH --array=0-11

module load maxwell mamba
. mamba-init
cd ~/training/point-cloud-diffusion
mamba activate caloi

cd /home/dayhallh/training/CC_ExpSpec/CaloCloudi
paths_to_time=(
"/data/dust/user/dayhallh/data/CaloClouds_diffusion/logs/2026_08_18__12_06_30/config.yaml"
"/data/dust/user/dayhallh/data/CaloClouds_diffusion/logs/2026_08_18__12_06_34/config.yaml"
"/data/dust/user/dayhallh/data/CaloClouds_diffusion/logs/2026_08_19__22_38_22/config.yaml"
"/data/dust/user/dayhallh/data/CaloClouds_diffusion/logs/2026_08_19__22_43_47/config.yaml"
"/data/dust/user/dayhallh/data/CaloClouds_diffusion/logs/2026_08_20__16_56_40/config.yaml"
"/data/dust/user/dayhallh/data/CaloClouds_diffusion/logs/2026_08_20__17_04_20/config.yaml"
)

# if SLURM_ARRAY_TASK_ID is not set, it will be 0
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    SLURM_ARRAY_TASK_ID=0
fi

# Use the paths_to_time array directly
configs=("${paths_to_time[@]}")
n_configs=${#configs[@]}
versions=("student" "teacher")
n_versions=${#versions[@]}
total_jobs=$(( n_configs * n_versions ))

# Check if task ID is within range
if [ "$SLURM_ARRAY_TASK_ID" -ge "$total_jobs" ]; then
    echo "SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_ID out of range (0-$((total_jobs-1)))" >&2
    exit 0
fi

# Compute config and version indices
config_index=$(( SLURM_ARRAY_TASK_ID / n_versions ))
version_index=$(( SLURM_ARRAY_TASK_ID % n_versions ))

config="${configs[$config_index]}"
version="${versions[$version_index]}"

# give the start and end times of the whole thing
time_start=$SECONDS
time_allocated=$((60*45))
python3 scripts/generate_timings.py $time_allocated $config $version
time_end=$SECONDS
echo Ran for $((time_end-time_start))
