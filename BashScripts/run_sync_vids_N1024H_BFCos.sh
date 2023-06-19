#!/bin/bash

# cd PostProcessing/; make; cd ..; 

## Set solver parameters
N=1024
T=200.000
trans_frac=0.75
save=500
num_procs=16

## Set the post parameter space
num_sects=(24)
kmin=0.00
c_radius_frac=(0.05 0.10 0.20 0.30)

## Command variables
input_dir="./Data/SectorSyncResults/Long/"
tag="Test"

num_plot_procs=25

## Run post processing and plotting 
for sec in "${num_sects[@]}"
do 
	for c_rad in "${c_radius_frac[@]}"
	do
		## Get solver tags and data directory
		solv_tag="$tag-S$sec-$c_rad"
		data_dir=$input_dir"NAV_AB4_FULL_N[$N,$N]_T[0.0,0.00025,"$T"]_NU[5e-20,1,4.0]_DRAG[0.1,1,0.0]_FORC[BODY_FORC_COS,2,1]_u0[RANDOM_ENRG]_TAG[$solv_tag]/"

		## Solver command
		post_command="mpirun -n $num_procs Solver/bin/solver -o $input_dir -n $N -n $N -s 0.00000 -e $T -T 1 -T $trans_frac -c 1 -c 0.900000 -h 0.00025000 -h 0 -v 5e-20 -v 1 -v 4.0 -d 0.1 -d 1 -d 0.0 -i RANDOM_ENRG -t $solv_tag -f BODY_FORC_COS -f 2 -f 1.000 -p $save"
		echo -e "\nCommand run:\n\t \033[1;36m $post_command \033[0m"
		$post_command  &
	done

	wait
done

wait
echo "Solvers Done!!"

## Run post processing and plotting 
for sec in "${num_sects[@]}"
do 
	for c_rad in "${c_radius_frac[@]}"
	do
		## Get solver tags and data directory
		solv_tag="$tag-S$sec-$c_rad"
		data_dir=$input_dir"NAV_AB4_FULL_N[$N,$N]_T[0.0,0.00025,"$T"]_NU[5e-20,1,4.0]_DRAG[0.1,1,0.0]_FORC[BODY_FORC_COS,2,1]_u0[RANDOM_ENRG]_TAG[$solv_tag]/"

		## Post command
		post_command="PostProcessing/bin/PostProcess_phase_sync -i $data_dir -o $data_dir -v 5e-20 -v 1 -v 4.0 -d 0.1 -d 1 -d 0.0 -f BODY_FORC_COS -f 2 -f 1.0 -t $tag -a $sec -a $sec -k $c_rad -k $kmin -p 1 -p 1"
		echo -e "\nCommand run:\n\t \033[1;36m $post_command \033[0m"
		$post_command &
	done

	wait
done

wait
echo "Post Done!!"

## Run post processing and plotting 
for sec in "${num_sects[@]}"
do 
	for c_rad in "${c_radius_frac[@]}"
	do
		## Get solver tags and data directory
		solv_tag="$tag-S$sec-$c_rad"
		data_dir=$input_dir"NAV_AB4_FULL_N[$N,$N]_T[0.0,0.00025,"$T"]_NU[5e-20,1,4.0]_DRAG[0.1,1,0.0]_FORC[BODY_FORC_COS,2,1]_u0[RANDOM_ENRG]_TAG[$solv_tag]/"

		## Plotting command
		plotting_command="python3 Plotting/plot_jet_sync.py -i $data_dir --triads=0 --phase_order --vid --plot --par -p $num_plot_procs -f PostProcessing_HDF_Data_THREADS[1,1]_SECTORS["$sec","$sec"]_KFRAC["$kmin","$c_rad"]_TAG[$tag].h5 --full=all -t $tag"
		echo -e "\nCommand run:\n\t \033[1;36m $plotting_command \033[0m"
		$plotting_command &
		
	done
	
	wait
done

wait 
echo "Plotting Done!" 
fg