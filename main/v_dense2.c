#include <stdio.h>
#include "../kernels.c"

int main(int argc, char *argv[]) {
    int i;
    int j;
    int k;
    float sum;
    int * shape_a = (int *) malloc(2 * sizeof(int));
    int * shape_b = (int *) malloc(2 * sizeof(int));
    shape_a[0] = atoi(argv[1]); shape_a[1] = atoi(argv[2]);
    shape_b[0] = atoi(argv[3]); shape_b[1] = atoi(argv[4]);
    int A = shape_a[0];
    int B = shape_a[1];
    int C = shape_b[1];    
    uint64_t start_cycles;
    uint64_t start_instret;
    uint64_t end_cycles;
    uint64_t end_instret;
    uint64_t cycles;
    uint64_t instret;

    float * a = (float * ) malloc(A*B * sizeof(float));
    float * b = (float * ) malloc(B*C * sizeof(float));
    float * c = (float * ) malloc(A*C * sizeof(float));

    // TODO you can remove this
    for(i = 0; i < A*B; i++)
        a[i] = (i + 1)%10;
    for(i = 0; i < B*C; i++)
        b[i] = (9 - i)%10;

    FILE * file = fopen("results.txt", "a");
    if (file == NULL) {
        perror("Error opening the file");
        return 1;
    }

    __asm__ __volatile__("rdinstret %0": "=r"(start_instret));
    __asm__ __volatile__("rdcycle %0": "=r"(start_cycles));
    dense2(a, b, c, shape_a, shape_b);
    __asm__ __volatile__("rdcycle %0": "=r"(end_cycles));
    __asm__ __volatile__("rdinstret %0": "=r"(end_instret));

    cycles = end_cycles - start_cycles;
    instret = end_instret - start_instret;

    free(a);
    free(b);
    free(c);

    fprintf(file, "%15s, %15lu, %15lu\n", "Vector dense2", cycles, instret);

    fclose(file);

    return 0;
}