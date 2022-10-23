#!/bin/bash -l
#SBATCH --job-name=2dns
#SBATCH -N 1
#SBATCH --ntasks-per-node 4
#SBATCH -t 00:20:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=16203944@ucdconnect.ie
cd $SLURM_SUBMIT_DIR
export FI_PROVIDER=verbs
module load hdf5/1.10.5
module load fftw/3.3.8


mpirun -n 4 Solver/bin/solver -o ./Data/Working/ -n 256 -n 256 -s 0.00000 -e 20.00000 -T 0 -c 1 -c 0.900000 -h 0.001000 -h 0 -v 0.0000100000 -v 0 -v 2.0 -d 0.200000 -d 0 -d -2.0 -i UNIF -t Kolo-Test-Run -f KOLM -f 4 -f 1.0 -p 500