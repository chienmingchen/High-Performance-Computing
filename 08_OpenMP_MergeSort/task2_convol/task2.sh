#!/usr/bin/env bash
#SBATCH -p wacc

#SBATCH -t 0-00:05:00
#SBATCH -J task2
#SBATCH -o output.out

#SBATCH --gres=gpu:1 -c 1
#SBATCH -N 1 
#SBATCH --mem=16G
#SBATCH --nodes=1 --cpus-per-task=20


g++ task2.cpp convolution.cpp -Wall -O3 -o task2 -fopenmp


export OMP_PLACES=threads
export OMP_PROC_BIND=close

./task2 1024 1
./task2 1024 2
./task2 1024 3
./task2 1024 4
./task2 1024 5
./task2 1024 6
./task2 1024 7
./task2 1024 8
./task2 1024 9
./task2 1024 10
./task2 1024 11
./task2 1024 12
./task2 1024 13
./task2 1024 14
./task2 1024 15
./task2 1024 16
./task2 1024 17
./task2 1024 18
./task2 1024 19
./task2 1024 20