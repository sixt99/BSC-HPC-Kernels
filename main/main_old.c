#include "kernels.c"

// SUM OF TWO VECTORS 
int main_vec(void) {
    int i;
    int j;
    int k;
    int repetitions = 5;
    int N;
    float * a;
    float * b;
    float * c;
    uint64_t start_cycles;
    uint64_t start_instret;
    uint64_t end_cycles;
    uint64_t end_instret;
    uint64_t cycles1;
    uint64_t cycles2;
    uint64_t instret1;
    uint64_t instret2;
    FILE * file = fopen("results.txt", "w");

    fprintf(file, "%15s %15s %15s %15s %15s %15s %15s\n", "N", "cycles1", "cycles2", "instret1", "instret2", "cyc1/cyc2", "inst1/inst2");

    for (j = 0; j < 100; j++) {

        cycles1 = UINT64_MAX;
        cycles2 = UINT64_MAX;
        instret1 = UINT64_MAX;
        instret2 = UINT64_MAX;
        for (k = 0; k < repetitions; k++) {

            float * a = (float * ) malloc(N * sizeof(float));
            float * b = (float * ) malloc(N * sizeof(float));
            float * c = (float * ) malloc(N * sizeof(float));

            srand(time(NULL));
            for (i = 0; i < N; i++) {
                a[i] = ((float) rand() / (float) RAND_MAX) * 100 - 50;
                b[i] = ((float) rand() / (float) RAND_MAX) * 100 - 50;
            }

            // SCALAR VERSION
            __asm__ __volatile__("rdinstret %0": "=r"(start_instret));
            __asm__ __volatile__("rdcycle %0": "=r"(start_cycles));

            scalar_vector_sum(a, b, c, N);

            __asm__ __volatile__("rdcycle %0": "=r"(end_cycles));
            __asm__ __volatile__("rdinstret %0": "=r"(end_instret));

            if (cycles1 > end_cycles - start_cycles) cycles1 = end_cycles - start_cycles;
            if (instret1 > end_instret - start_instret) instret1 = end_instret - start_instret;
            //fprintf(file, "SCALAR");	
            //fprintf(file, "we got %.lu %.lu\n", end_cycles  - start_cycles, end_instret - start_instret);
            //fprintf(file, "we set %.lu %.lu\n", cycles1, instret1);

            // VECTOR VERSION
            __asm__ __volatile__("rdinstret %0": "=r"(start_instret));
            __asm__ __volatile__("rdcycle %0": "=r"(start_cycles));

            vector_sum(a, b, c, N);

            __asm__ __volatile__("rdcycle %0": "=r"(end_cycles));
            __asm__ __volatile__("rdinstret %0": "=r"(end_instret));

            if (cycles2 > end_cycles - start_cycles) cycles2 = end_cycles - start_cycles;
            if (instret2 > end_instret - start_instret) instret2 = end_instret - start_instret;
            //fprintf(file, "VECTOR");
            //fprintf(file, "we got %.lu %.lu\n", end_cycles  - start_cycles, end_instret - start_instret);
            //fprintf(file, "we set %.lu %.lu\n", cycles2, instret2);

            free(a);
            free(b);
            free(c);

        }
        fprintf(file, "%15d %15.lu %15.lu %15.lu %15.lu %15.2f %15.2f\n", N, cycles1, cycles2, instret1, instret2, (float) cycles1 / cycles2, (float) instret1 / instret2);
    }

    return 0;
}

// RELU
int main_relu(void) {
    int i;
    int j;
    int l;
    int N = 3;
    int M = 3;

    uint64_t start_cycles;
    uint64_t start_instret;
    uint64_t end_cycles;
    uint64_t end_instret;

    FILE * file = fopen("results.txt", "w");

    int shapes1[21][3] = {
        {
            1,
            1,
            120
        },
        {
            5,
            5,
            16
        },
        {
            1,
            32,
            32
        },
        {
            14,
            14,
            6
        },
        {
            10,
            10,
            16
        },
        {
            28,
            28,
            6
        },
        {
            128,
            14,
            14
        },
        {
            512,
            7,
            7
        },
        {
            1024,
            7,
            7
        },
        {
            256,
            14,
            14
        },
        {
            64,
            28,
            28
        },
        {
            128,
            28,
            28
        },
        {
            2048,
            7,
            7
        },
        {
            32,
            56,
            56
        },
        {
            512,
            14,
            14
        },
        {
            1024,
            14,
            14
        },
        {
            256,
            28,
            28
        },
        {
            64,
            56,
            56
        },
        {
            32,
            112,
            112
        },
        {
            512,
            28,
            28
        },
        {
            256,
            56,
            56
        }
    };

    int shapes[3][3] = {
        {
            256,
            56,
            56
        },
        {
            256,
            56,
            56
        },
        {
            256,
            56,
            56
        }
    };

    int * shape = (int * ) malloc(M * sizeof(int));
    float alpha = 0.5;

    fprintf(file, "%15s %15s %15s %15s %15s %15s %15s\n", "shape", "cycles1", "cycles2", "instret1", "instret2", "cyc1/cyc2", "inst1/inst2");

    for (i = 0; i < N; i++) {

        // Take one row of shapes
        for (j = 0; j < M; j++)
            shape[j] = shapes[i][j];

        // Multiply shapes
        l = 1;
        for (j = 0; j < M; j++)
            l *= shape[j];

        float * T = (float * ) malloc(l * sizeof(float));
        float * D = (float * ) malloc(l * sizeof(float));

        // Initialize tensor
        srand(time(NULL));
        for (j = 0; j < l; j++)
            T[j] = ((float) rand() / (float) RAND_MAX) * 100 - 50;

        // SCALAR VERSION
        __asm__ __volatile__("rdinstret %0": "=r"(start_instret));
        __asm__ __volatile__("rdcycle %0": "=r"(start_cycles));

        scalar_relu(T, D, alpha, shape, M);

        __asm__ __volatile__("rdcycle %0": "=r"(end_cycles));
        __asm__ __volatile__("rdinstret %0": "=r"(end_instret));

        uint64_t cycles1 = end_cycles - start_cycles;
        uint64_t instret1 = end_instret - start_instret;

        // VECTOR VERSION
        __asm__ __volatile__("rdinstret %0": "=r"(start_instret));
        __asm__ __volatile__("rdcycle %0": "=r"(start_cycles));

        relu(T, D, alpha, shape, M);

        __asm__ __volatile__("rdcycle %0": "=r"(end_cycles));
        __asm__ __volatile__("rdinstret %0": "=r"(end_instret));

        free(T);
        free(D);

        uint64_t cycles2 = end_cycles - start_cycles;
        uint64_t instret2 = end_instret - start_instret;

        fprintf(file, "{%d, ", shape[0]);
        for (j = 1; j < M - 1; j++)
            fprintf(file, "%d, ", shape[j]);
        fprintf(file, "%d}\t", shape[M - 1]);

        fprintf(file, "%15.lu %15.lu %15.lu %15.lu %15.2f %15.2f\n", cycles1, cycles2, instret1, instret2, (float) cycles1 / cycles2, (float) instret1 / instret2);
    }

    return 0;
}

// Set the name of this function to main to run
int main_axpy(void) {
    int i;
    int j;
    int N;
    float c;

    uint64_t start_cycles;
    uint64_t start_instret;
    uint64_t end_cycles;
    uint64_t end_instret;

    FILE * file = fopen("results.txt", "w");
    fprintf(file, "%15s %15s %15s %15s %15s %15s %15s %15s %15s %15s %15s\n", "N", "cycles1", "cycles2", "cycles3", "instret1", "instret2", "instret3", "cyc1/cyc2", "inst1/inst2", "cyc1/cyc3", "inst1/inst3");

    //for(N=100; N<= 1000; N+=100){
    for (j = 0; j < 4; j++) {
        N = 1000;
        float * a = (float * ) malloc(N * sizeof(float));
        float * b = (float * ) malloc(N * sizeof(float));

        srand(time(NULL));
        for (i = 0; i < N; i++) {
            a[i] = ((float) rand() / (float) RAND_MAX) * 100 - 50;
            b[i] = ((float) rand() / (float) RAND_MAX) * 100 - 50;
        }

        // SCALAR VERSION
        __asm__ __volatile__("rdinstret %0": "=r"(start_instret));
        __asm__ __volatile__("rdcycle %0": "=r"(start_cycles));

        scalar_axpy(a, b, & c, N);

        __asm__ __volatile__("rdcycle %0": "=r"(end_cycles));
        __asm__ __volatile__("rdinstret %0": "=r"(end_instret));

        uint64_t cycles1 = end_cycles - start_cycles;
        uint64_t instret1 = end_instret - start_instret;

        // VECTOR VERSION
        __asm__ __volatile__("rdinstret %0": "=r"(start_instret));
        __asm__ __volatile__("rdcycle %0": "=r"(start_cycles));

        axpy(a, b, & c, N);

        __asm__ __volatile__("rdcycle %0": "=r"(end_cycles));
        __asm__ __volatile__("rdinstret %0": "=r"(end_instret));

        uint64_t cycles2 = end_cycles - start_cycles;
        uint64_t instret2 = end_instret - start_instret;

        // IMPROVED VERSION
        __asm__ __volatile__("rdinstret %0": "=r"(start_instret));
        __asm__ __volatile__("rdcycle %0": "=r"(start_cycles));

        efficient_axpy(a, b, & c, N);

        __asm__ __volatile__("rdcycle %0": "=r"(end_cycles));
        __asm__ __volatile__("rdinstret %0": "=r"(end_instret));

        uint64_t cycles3 = end_cycles - start_cycles;
        uint64_t instret3 = end_instret - start_instret;

        free(a);
        free(b);

        fprintf(file, "%15d %15.lu %15.lu %15.lu %15.lu %15.lu %15.lu %15.2f %15.2f %15.2f %15.2f\n", N, cycles1, cycles2, cycles3, instret1, instret2, instret3, (float) cycles1 / cycles2, (float) instret1 / instret2, (float) cycles1 / cycles3, (float) instret1 / instret3);

    }

    return 0;
}

// Set the name of this function to main to run
int main_sum(void) {
    int i;
    int j;
    int ndims = 1;
    int ntensors = 5;
    int * shape = (int * ) malloc(ndims * sizeof(int));
    shape[0] = 200;
    uint64_t start_cycles;
    uint64_t start_instret;
    uint64_t end_cycles;
    uint64_t end_instret;

    float * out = (float * ) malloc(shape[0] * sizeof(float)); // Shapes should be multiplied if more than one

    // Memory for tensors
    float ** tensors = (float ** ) malloc(ntensors * sizeof(float * ));
    if (tensors == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    for (int i = 0; i < ntensors; i++) {
        tensors[i] = (float * ) malloc(shape[0] * sizeof(float));
        if (tensors[i] == NULL) {
            printf("Memory allocation failed\n");
            return 1;
        }
    }

    // Initialize tensors
    for (i = 0; i < ntensors; i++) {
        for (j = 0; j < shape[0]; j++) {
            tensors[i][j] = (i + 1) * (j + 1);
        }
    }

    __asm__ __volatile__("rdinstret %0": "=r"(start_instret));
    __asm__ __volatile__("rdcycle %0": "=r"(start_cycles));

    // Sum of tensors
    sum(tensors, out, shape, ndims, ntensors);

    __asm__ __volatile__("rdcycle %0": "=r"(end_cycles));
    __asm__ __volatile__("rdinstret %0": "=r"(end_instret));

    uint64_t cycles = end_cycles - start_cycles;
    uint64_t instret = end_instret - start_instret;

    // Print result
    for (i = 0; i < shape[0]; i++) {
        printf("%.2f ", out[i]);
    }

    printf("\nINSTRUCTIONS AND CYCLES: %lu %lu\n", instret, cycles);

    return 0;
}

int main_(void) {
    int i;
    int N = 1000;
    float * a = (float * ) malloc(N * sizeof(float));
    float * b = (float * ) malloc(N * sizeof(float));
    float c;
    FILE * file = fopen("results.txt", "w");
    uint64_t start_cycles;
    uint64_t start_instret;
    uint64_t end_cycles;
    uint64_t end_instret;

    for (i = 0; i < N; i++) {
        a[i] = 1;
        b[i] = 2;
    }

    __asm__ __volatile__("rdinstret %0": "=r"(start_instret));
    __asm__ __volatile__("rdcycle %0": "=r"(start_cycles));

    axpy(a, b, & c, N);

    __asm__ __volatile__("rdcycle %0": "=r"(end_cycles));
    __asm__ __volatile__("rdinstret %0": "=r"(end_instret));

    uint64_t cycles = end_cycles - start_cycles;
    uint64_t instret = end_instret - start_instret;

    fprintf(file, "%f\n", c);
    fprintf(file, "%lu %lu\n", cycles, instret);

    return 0;
}