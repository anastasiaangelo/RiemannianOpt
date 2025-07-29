#!/bin/bash
#SBATCH --job-name=ROPT-FWD-HYB
#SBATCH --array=0          # Adjust to match your param list
#SBATCH --cpus-per-task=1
#SBATCH --time=100:00:00      # Each sweep should now be shorter
#SBATCH --output=A%j.o
#SBATCH --error=A%j.e

MAIN_JOB_ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
HOMEDIR=$(pwd)
RUNDIR="$HOMEDIR/A$MAIN_JOB_ID"
mkdir -p "$RUNDIR"

TASK_RUNDIR="$RUNDIR/run_$SLURM_ARRAY_TASK_ID"
mkdir -p "$TASK_RUNDIR"
cp "$HOMEDIR/simulations_job.py" "$TASK_RUNDIR/"

cd "$TASK_RUNDIR" || exit

source /home/b/aag/ropt-aqc/ropt-cluster-venv/bin/activate

if [[ $SLURM_ARRAY_TASK_ID -eq 0 ]]; then
    COMMIT=$(git rev-parse HEAD)
    echo "$COMMIT" > "$RUNDIR/commit_hash.txt"
else
    COMMIT=$(cat "$RUNDIR/commit_hash.txt")
fi

python3 -u simulations_job.py "$COMMIT" "$SLURM_ARRAY_TASK_ID" hybrid_forward \
    > "$TASK_RUNDIR/task_output.out" 2> "$TASK_RUNDIR/task_error.err"
