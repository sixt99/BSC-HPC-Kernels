#include <stdio.h>
#include "../kernels.c"

int main(int argc, char *argv[]) {
    int i;
    int N = atoi(argv[1]);
    float * a = (float * ) malloc(N * sizeof(float));
    float * b = (float * ) malloc(N * sizeof(float));
    float * c = (float * ) malloc(N * sizeof(float));
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
    
    __asm__ __volatile__("rdinstret %0": "=r"(start_instret));
    __asm__ __volatile__("rdcycle %0": "=r"(start_cycles));
    scalar_vector_sum(a, b, c, N);
    __asm__ __volatile__("rdcycle %0": "=r"(end_cycles));
    __asm__ __volatile__("rdinstret %0": "=r"(end_instret));

    cycles = end_cycles - start_cycles;
    instret = end_instret - start_instret;

    fprintf(file, "%15s, %15d, %15lu, %15lu\n", "Sc. vector sum", N, cycles, instret);

    free(a);
    free(b);
    free(c);

    fclose(file);

    return 0;
}
