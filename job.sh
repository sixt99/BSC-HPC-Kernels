#!/bin/bash
#SBATCH --job-name="test_job"
#SBATCH --output=traces/serial_%j.out
#SBATCH --error=traces/serial_%j.err
#SBATCH --ntasks=1
#SBATCH --partition=arriesgado-jammy
#SBATCH --time=00:02:00

# current_datetime=$(date +"%Y-%m-%d %H:%M:%S")
# printf "\nVEHAVE run $current_datetime:\n" >> results.txt

module load llvm/EPI-0.7-development 
clang -mepi -O0 main/s_vector_sum.c -o s_vector_sum
clang -mepi -O0 main/v_vector_sum.c -o v_vector_sum
clang -mepi -O0 main/s_relu.c -o s_relu
clang -mepi -O0 main/v_relu.c -o v_relu
clang -mepi -O0 main/s_axpy.c -o s_axpy
clang -mepi -O0 main/v_axpy.c -o v_axpy
clang -mepi -O0 main/o_axpy.c -o o_axpy
clang -mepi -O0 main/s_sum.c -o s_sum
clang -mepi -O0 main/v_sum.c -o v_sum
clang -mepi -O0 main/s_dense.c -o s_dense
clang -mepi -O0 main/v_dense.c -o v_dense
clang -mepi -O0 main/v_dense2.c -o v_dense2
clang -mepi -O0 main/s_softmax.c -o s_softmax
#clang -mepi -O0 main/v_softmax.c -o v_softmax

#LD_PRELOAD=/apps/riscv/vehave/EPI-0.7/development/lib64/libvehave.so ./s_vector_sum
#LD_PRELOAD=/apps/riscv/vehave/EPI-0.7/development/lib64/libvehave.so ./v_vector_sum
#LD_PRELOAD=/apps/riscv/vehave/EPI-0.7/development/lib64/libvehave.so ./s_relu 3 5 4
#LD_PRELOAD=/apps/riscv/vehave/EPI-0.7/development/lib64/libvehave.so ./v_relu 3 5 4
#LD_PRELOAD=/apps/riscv/vehave/EPI-0.7/development/lib64/libvehave.so ./s_axpy
#LD_PRELOAD=/apps/riscv/vehave/EPI-0.7/development/lib64/libvehave.so ./v_axpy
#LD_PRELOAD=/apps/riscv/vehave/EPI-0.7/development/lib64/libvehave.so ./o_axpy
#LD_PRELOAD=/apps/riscv/vehave/EPI-0.7/development/lib64/libvehave.so ./s_sum 3 5 4
#LD_PRELOAD=/apps/riscv/vehave/EPI-0.7/development/lib64/libvehave.so ./v_sum 3 5 4
#LD_PRELOAD=/apps/riscv/vehave/EPI-0.7/development/lib64/libvehave.so ./s_dense 10 600 600 10
#LD_PRELOAD=/apps/riscv/vehave/EPI-0.7/development/lib64/libvehave.so ./v_dense 10 600 600 10
#LD_PRELOAD=/apps/riscv/vehave/EPI-0.7/development/lib64/libvehave.so ./v_dense2 10 600 600 10
LD_PRELOAD=/apps/riscv/vehave/EPI-0.7/development/lib64/libvehave.so ./s_softmax 3 5
#LD_PRELOAD=/apps/riscv/vehave/EPI-0.7/development/lib64/libvehave.so ./v_softmax 3 5