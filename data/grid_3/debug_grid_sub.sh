#!/bin/bash
#SBATCH --job-name=star_grid
#SBATCH --output=star_grid_%A_%a.out # %A: master job ID, %a: array tasks ID.
#SBATCH --array=0-5
#SBATCH -N 1   # node count. OpenMP requires 1.
#SBATCH --ntasks-per-node=1  # core count
#SBATCH -t 3:59:00 # 5min gets test queue.
# sends mail when process begins, and
# when it ends. Make sure you define your email
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=lbouma@princeton.edu

# Set variables
export MESA_DIR="/home/lbouma/software/mesa/mesa-mist"
export OMP_NUM_THREADS=1
export MESA_BASE="/home/lbouma/software/mesa/base"
export MESA_INLIST="$MESA_BASE/inlist"
export MESA_RUN="/home/lbouma/software/mesa/run-grids"

# cd to folder
rundir="$(find $MESA_RUN -type d -regex $MESA_RUN"/[$SLURM_ARRAY_TASK_ID]*_.*")"
cd rundir

# note list of all folders is:
# find $MESA_RUN -type d -regex $MESA_RUN"/[0-9]*_.*"

# Run MESA (with MIST params)!

#cd /home/lbouma/software/mesa/tides-project
#srun ./run_mesa_serial.py
#if [ -d "$HOME/software/mesa" ]; then
#  export MESASDK_ROOT=$HOME/software/mesa/mesasdk-r7503
#  source $MESASDK_ROOT/bin/mesasdk_init.sh
#  if [[ "$(hostname)" == adroit* ]]; then
#  fi
#fi
