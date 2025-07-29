#!/bin/bash
#SBATCH --job-name=PXP_sweep
#SBATCH --array=1-24%4       # 24 tasks, 4 running in parallel
#SBATCH --cpus-per-task=1
#SBATCH --output=A%j.o               # Single output file for the job
#SBATCH --error=A%j.e                # Single error file for the job
#SBATCH --time=48:00:00

# Get the correct job ID (ensuring it’s the same for all tasks in the array)
MAIN_JOB_ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}  # Ensures a single job ID across tasks

# Define home and job-wide run directory (ensuring a single job folder)
HOMEDIR=$(pwd)
RUNDIR="$HOMEDIR/A$MAIN_JOB_ID"  # All tasks share this directory

# Ensure the main job directory exists (only created once)
mkdir -p "$RUNDIR"

# Each task creates its own unique subdirectory inside the job directory
TASK_RUNDIR="$RUNDIR/run_$SLURM_ARRAY_TASK_ID"
mkdir -p "$TASK_RUNDIR"

# Copy input files into the task directory
cp "$HOMEDIR/simulations_array.py" "$TASK_RUNDIR/"

# Move to the task-specific directory
cd "$TASK_RUNDIR" || exit

# Activate environment
source /home/b/aag/ropt_aqc/ropt-cluster-venv/bin/activate

# Get the git commit hash (only once, stored in the main job folder)
if [[ $SLURM_ARRAY_TASK_ID -eq 1 ]]; then
    COMMIT=$(git rev-parse HEAD)
    echo "Running script with commit: $COMMIT"
    echo "$COMMIT" > "$RUNDIR/commit_hash.txt"  # Save commit hash for reference
else
    COMMIT=$(cat "$RUNDIR/commit_hash.txt")  # Read stored commit
fi

# Run the simulation and save JSON output inside the task folder
python3 -u simulations_array.py "$COMMIT" "$SLURM_ARRAY_TASK_ID" \
    > "$TASK_RUNDIR/task_output.out" 2> "$TASK_RUNDIR/task_error.err"
