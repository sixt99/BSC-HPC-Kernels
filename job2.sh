#!/bin/bash
#SBATCH --partition=fpga-sdv
#SBATCH --nodes=1
#SBATCH --time=2:00:00

current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
printf "\nRun $current_datetime:\n" >> results.txt

X86_HOST="`hostname`"
SDV_HOST="fpga-sdv-`echo ${X86_HOST} | cut -d '-' -f 2`"

# VECTOR SUM
printf "kind,N,cycles,instructions\n"

for ((N = 20000; N <= 500000; N+=20000)); do
  ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_vector_sum $N"
done
for ((N = 20000; N <= 500000; N+=20000)); do
  ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_vector_sum $N"
done

# RELU
printf "kind,shape,size,cycles,instructions\n"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 1 12O"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 5 5 16"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 32 32"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 14 14 6"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 10 10 16"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 28 28 6"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 128 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 512 7 7"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 1024 7 7"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 256 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 64 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 128 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 2048 7 7"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 32 56 56"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 512 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 1024 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 256 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 64 56 56"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 32 112 112"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 512 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_relu 1 256 56 56"

ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 1 12O"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 5 5 16"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 32 32"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 14 14 6"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 10 10 16"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 28 28 6"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 128 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 512 7 7"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 1024 7 7"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 256 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 64 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 128 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 2048 7 7"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 32 56 56"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 512 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 1024 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 256 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 64 56 56"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 32 112 112"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 512 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_relu 1 256 56 56"

# AXPY
printf "kind,N,cycles,instruction\n"
for ((N = 20000; N <= 500000; N+=20000)); do
  ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_axpy $N"
done
for ((N = 20000; N <= 500000; N+=20000)); do
  ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_axpy $N"
done
for ((N = 20000; N <= 500000; N+=20000)); do
  ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./o_axpy $N"
done

# SUM
printf "kind,shape,size,ntensors,cycles,instructions\n"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 1 12O"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 5 5 16"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 32 32"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 14 14 6"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 10 10 16"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 28 28 6"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 128 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 512 7 7"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 1024 7 7"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 256 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 64 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 128 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 2048 7 7"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 32 56 56"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 512 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 1024 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 256 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 64 56 56"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 32 112 112"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 512 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./s_sum 1 256 56 56"

ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 1 12O"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 5 5 16"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 32 32"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 14 14 6"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 10 10 16"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 28 28 6"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 128 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 512 7 7"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 1024 7 7"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 256 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 64 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 128 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 2048 7 7"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 32 56 56"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 512 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 1024 14 14"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 256 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 64 56 56"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 32 112 112"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 512 28 28"
ssh -o StrictHostKeyChecking=no ${SDV_HOST} "./v_sum 1 256 56 56"