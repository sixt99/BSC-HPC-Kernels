#include <stdio.h>
#include "kernels.c"

int main(int argc, char *argv[]) {
    int i;
    int N = atoi(argv[1]);
    float c;
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
    
    float * a = (float * ) malloc(N * sizeof(float));
    float * b = (float * ) malloc(N * sizeof(float));

    srand(time(NULL));
    for (i = 0; i < N; i++) {
        a[i] = ((float) rand() / (float) RAND_MAX) * 100 - 50;
        b[i] = ((float) rand() / (float) RAND_MAX) * 100 - 50;
    }

    __asm__ __volatile__("rdinstret %0": "=r"(start_instret));
    __asm__ __volatile__("rdcycle %0": "=r"(start_cycles));
    axpy(a, b, & c, N);
    __asm__ __volatile__("rdcycle %0": "=r"(end_cycles));
    __asm__ __volatile__("rdinstret %0": "=r"(end_instret));

    cycles = end_cycles - start_cycles;
    instret = end_instret - start_instret;

    free(a);
    free(b);

    fprintf(file, "%15s, %15d, %15lu, %15lu\n", "Vector axpy", N, cycles, instret);

    fclose(file);

    return 0;
}