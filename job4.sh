#!/bin/bash
#SBATCH --job-name="test_job"
#SBATCH --output=traces/serial_%j.out
#SBATCH --error=traces/serial_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=arriesgado-jammy
#SBATCH --time=00:02:00

clang -mepi -O0 main/s_softmax.c -o s_softmax
LD_PRELOAD=/apps/riscv/vehave/EPI-0.7/development/lib64/libvehave.so ./s_softmax 3 5