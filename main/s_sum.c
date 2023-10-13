#include <stdio.h>
#include <string.h>
#include "kernels.c"

int main(int argc, char *argv[]) {
    int i;
    int j;
    int l;
    float alpha = 0.5;
    int ndims = argc - 1;
    int * shape = (int * ) malloc(ndims * sizeof(int));
    int ntensors = 200;
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
    for (i = 0; i < ndims; i++)
        shape[i] = atoi(argv[i + 1]);

    // Create a string describing the shape
    char shape_str[100] = "{";
    for (i = 0; i < ndims; i++) {
        char str[20];
        sprintf(str, "%d", shape[i]);
        strcat(shape_str, str);
        if (i < ndims - 1) {
            strcat(shape_str, ",");
        }
    }
    strcat(shape_str, "}");

    // Multiply shapes
    l = 1;
    for (i = 0; i < ndims; i++)
        l *= shape[i];

    // Memory for tensors
    float ** tensors = (float ** ) malloc(ntensors * sizeof(float * ));
    if (tensors == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    for (int i = 0; i < ntensors; i++) {
        tensors[i] = (float * ) malloc(l * sizeof(float));
        if (tensors[i] == NULL) {
            printf("Memory allocation failed\n");
            return 1;
        }
    }
    
    float * out = (float * ) malloc(l * sizeof(float));

    srand(time(NULL));
    for (j = 0; j < l; j++)
    for(i = 0; i < ntensors; i++)
        tensors[i][j] = ((float) rand() / (float) RAND_MAX) * 100 - 50;

    __asm__ __volatile__("rdinstret %0": "=r"(start_instret));
    __asm__ __volatile__("rdcycle %0": "=r"(start_cycles));
    scalar_sum(tensors, out, shape, ndims, ntensors);
    __asm__ __volatile__("rdcycle %0": "=r"(end_cycles));
    __asm__ __volatile__("rdinstret %0": "=r"(end_instret));

    cycles = end_cycles - start_cycles;
    instret = end_instret - start_instret;

    for(i=0; i<ntensors; i++)
        free(tensors[i]);
    free(tensors);
    free(out);

    fprintf(file, "%15s, %15s, %15d, %15d, %15lu, %15lu\n", "Scalar sum", shape_str, l, ntensors, cycles, instret);

    fclose(file);

    return 0;
}