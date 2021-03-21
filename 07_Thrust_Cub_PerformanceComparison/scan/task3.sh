#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -t 0-00:20:00
#SBATCH -J task3
#SBATCH -o task3.out -e task3.out
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16G

module load cuda

nvcc task3.cu count.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -o task3 

for i in {5..24}
do
  N=$((2**$i))
  ./task3 $N
done


sudo chown -R james:uwm-group212 /mnt/data/