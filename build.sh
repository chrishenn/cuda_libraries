#!/bin/bash

# project dir
proj_dir=$(pwd)

# return true cores = logical-cores/2
n_proc=$(lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l)

# We will not clean build a lib unless the build dir is not present in the lib folder, or the -c option is passed in
clean_build_flag=0

# Display Help
Help() {
	echo "Syntax: ./install.sh [-c|h]"
	echo "options:"
	echo "c     force a clean build for all libs in the cuda_lib folder"
	echo "h     Print this Help."

	echo "
    This script will attempt to use cmake and make to build a library for each folder in project_dir/cuda_lib.
    The default behavior is to carry out 'dirty' builds, calling solely 'make' when possible.
    If the -c option is passed to this script, or a given build folder is not present, then it will attempt a 'clean' build.
    A clean build will delete and recreate any existing build folder if needed, and will call 'cmake ..', followed by 'make' in the new build folder.
    The number of true cores available according to nproc will be used to build via 'make -j n_cores'.
    "
}

# Get cmd options
while getopts ":hc" option; do

	case $option in

	c) # clean build
		clean_build_flag=1 ;;

	h) # display Help
		Help
		exit
		;;

	\?) # Invalid option
		echo "Error: Invalid option. Run './install.sh -h' for help"
		exit
		;;
	esac
done

# build a cuda lib in each subfolder of project_dir/cuda_lib/.
# Do a clean build if the -c option is passed, or if the build dir is missing for a given cuda lib.
if [ "$(ls -d cuda_lib/*/)" ]; then

	echo "Detected $n_proc available cores; building with $n_proc make threads"

	for cuda_library_dir in $(ls -d cuda_lib/*/); do

		if [ -d "$cuda_library_dir" ]; then

			cd $cuda_library_dir
			echo "Working: build a cuda lib in $(pwd)"

			if [ -d "./build" ]; then
				# build dir is present; only do clean build if -c option passed
				if [ $clean_build_flag = 1 ]; then
					echo -e "Working: clean build for lib at $cuda_library_dir \n"
					rm -rf "./build/"
					mkdir "./build"
					cd "./build"
					cmake ..
					make -j $n_proc
				else
					echo -e "Working: dirty build for lib at $cuda_library_dir \n"
					cd "./build"
					make -j $n_proc
				fi

			else
				# build dir is missing; do full build
				echo -e "Working: clean build for lib at $cuda_library_dir \n"
				mkdir "./build"
				cd "./build"
				cmake ..
				make -j $n_proc
			fi

			cd $proj_dir
			echo -e "\n"
		else
			echo "no folders found in project_dir/cuda_lib/; no cuda libs built"
		fi
	done
fi
