#include <stdio.h>
#include <string.h>
#include "kernels.c"

int main(int argc, char *argv[]) {
    int i;
    int l;
    float alpha = 0.5;
    int M = argc - 1;
    int * shape = (int * ) malloc(M * sizeof(int));
    uint64_t start_cycles;
    uint64_t start_instret;
    uint64_t end_cycles;
    uint64_t end_instret;
    uint64_t cycles;
    uint64_t instret;

    FILE * file = fopen("results.txt", "a");
    if (file == NULL) {
        perror("Error opening the file");
        return 1;
    }

    // Gather numbers given in script
    for (i = 0; i < M; i++)
        shape[i] = atoi(argv[i + 1]);

    // Create a string describing the shape
    char shape_str[100] = "{";
    for (i = 0; i < M; i++) {
        char str[20];
        sprintf(str, "%d", shape[i]);
        strcat(shape_str, str);
        if (i < M - 1) {
            strcat(shape_str, ",");
        }
    }
    strcat(shape_str, "}");

    // Multiply shapes
    l = 1;
    for (i = 0; i < M; i++)
        l *= shape[i];
        
    float * T = (float * ) malloc(l * sizeof(float));
    float * D = (float * ) malloc(l * sizeof(float));

    __asm__ __volatile__("rdinstret %0": "=r"(start_instret));
    __asm__ __volatile__("rdcycle %0": "=r"(start_cycles));
    scalar_relu(T, D, alpha, shape, M);
    __asm__ __volatile__("rdcycle %0": "=r"(end_cycles));
    __asm__ __volatile__("rdinstret %0": "=r"(end_instret));

    cycles = end_cycles - start_cycles;
    instret = end_instret - start_instret;

    free(T);
    free(D);

    fprintf(file, "%15s, %15s, %15d, %15lu, %15lu\n", "Scalar RELU", shape_str, l, cycles, instret);

    fclose(file);

    return 0;
}