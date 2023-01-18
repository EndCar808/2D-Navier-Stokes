#!/bin/bash

cd PostProcessing/; make; cd ..;

dump_dir="/home/enda/PhD/2D-Navier-Stokes/Data/Testing/ParStrFunc/ParallelRunsDump/*"
rm $dump_dir

## Run Script to produce the post data
post_data_cmd="python3 run_solver.py -i InitFiles/TestParallelStrFunc/par_str_func_data.ini"
echo -e "\nGathering Post Data:\n\t \033[1;36m $post_data_cmd \033[0m"
$post_data_cmd

## Run script to compare and plot data
num_threads="5"
num_n="8"
for i in {6..8}
do
	nn=$((2**i));
  plot_script="Plotting/plot_par_str_func_test.py";
  data_dir="Data/Testing/ParStrFunc/SIM_DATA_NAVIER_AB4CN_FULL_N[$nn,$nn]_T[0.0,0.001,100.000]_NU[5e-10,1,4.0]_DRAG[0.1,0.1,1,0.0]_CFL[0.90]_FORC[BODY_FORC,2,1]_u0[RANDOM]_TAG[ParStrFuncTest]/";
  compare_data_cmd="python3 $plot_script -i $data_dir -f PostProcessing_HDF_Data_THREADS[1,1]_SECTORS[24,24]_KFRAC[0.50]_TAG[ParStrFunc-Test-Base].h5 -p $num_threads";
  echo -e "\nComparing Str Func Data:\n\t \033[1;36m $compare_data_cmd \033[0m";
  $compare_data_cmd
done
## Run execution scaling
out_dir_file="/home/enda/PhD/2D-Navier-Stokes/Data/Testing/ParStrFunc/ParallelRunsDump/par_run_post_output*"
speed_up_script="Plotting/plot_par_strfunc_speed_up.py"
plot_speed_up_cmd="python3 $speed_up_script -f $out_dir_file -p $num_threads -n $num_n"
echo -e "\nPar Str Func Speedup:\n\t \033[1;36m $plot_speed_up_cmd \033[0m"
$plot_speed_up_cmd
