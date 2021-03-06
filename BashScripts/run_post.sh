#!/bin/bash

cd PostProcessing/; make; cd ..; 

## Set the parameter space
num_sects=(24 48)
c_radius_frac=(0.50 0.60)

## Command variables
data_dir="Data/Tmp/SIM_DATA_NAVIER_RK4_FULL_N[128,128]_T[0-500]_NU[0.000000]_CFL[0.80]_u0[DECAY_TURB_ALT]_TAG[Decay-Test-Alt]/"
tag="FULL-PHASE-SYNC"
nu=3.125e-07

## Run post processing and plotting 
for i in "${num_sects[@]}"
do 
	for j in "${c_radius_frac[@]}"
	do 
		solver_command="PostProcessing/bin/main -i $data_dir -o $data_dir -a "$i" -a 1 -k "$j" -v $nu -t $tag"
		echo -e "\nCommand run:\n\t \033[1;36m $solver_command \033[0m"
		$solver_command

		plotting_command="python3 Plotting/plot_jet_sync.py -i $data_dir --triads=0 --plot --vid --par -f PostProcessing_HDF_Data_SECTORS["$i"]_KFRAC["$j"]_TAG[$tag].h5 --full=sec -t $tag"
		echo -e "\nCommand run:\n\t \033[1;36m $plotting_command \033[0m"
		$plotting_command
	done
done
