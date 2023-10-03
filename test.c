#include <stdio.h>

#include <stdlib.h>

#include <stdint.h>

#include <time.h>

void scalar_vector_sum(float * , float * , float * , int);
void vector_sum(float * , float * , float * , int);

void triad(float[][32], float[][32], float[][32], int, int);
void efficient_triad(float * , float * , float * , int, int);

void scalar_relu(float * , float * , float, int * , int);
void relu(float * , float * , float, int * , int);

void scalar_axpy(float * , float * , float * , int);
void axpy(float * , float * , float * , int);
void efficient_axpy(float * , float * , float * , int);

void sum(float ** , float * , int * , int, int);
void unrolled2_sum(float ** , float * , int * , int, int);

// ------------------------------------
// ---------- MAIN FUNCTIONS ----------
// ------------------------------------

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
    shape[0] = 129;
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

// ------------------------------------
// ------------ FUNCTIONS  ------------
// ------------------------------------

// Sum element-wise two vectors
void scalar_vector_sum(float * a, float * b, float * c, int N) {
    int i;
    for (i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }

    return;
}

// Sum element-wise two vectors using vectorized instructions
void vector_sum(float * a, float * b, float * c, int N) {
    int i;
    int gvl;
    __epi_2xf32 va;
    __epi_2xf32 vb;
    __epi_2xf32 vc;

    for (i = 0; i < N; i += gvl) {
        gvl = __builtin_epi_vsetvl(N - i, __epi_e32, __epi_m1);
        va = __builtin_epi_vload_2xf32( & a[i], gvl);
        vb = __builtin_epi_vload_2xf32( & b[i], gvl);
        vc = __builtin_epi_vfadd_2xf32(va, vb, gvl);
        __builtin_epi_vstore_2xf32( & c[i], vc, gvl);
    }

    return;
}

// Sum element-wise two matrices
void triad(float A[][32], float B[][32], float C[][32], int M, int N) {
    int i;
    int j;
    int gvl;
    __epi_2xf32 va;
    __epi_2xf32 vb;
    __epi_2xf32 vc;

    for (i = 0; i < M; i++) {
        printf("i=%d\n", i);
        for (j = 0; j < N; j += gvl) {
            printf("j=%d\n", j);
            gvl = __builtin_epi_vsetvl(N - j, __epi_e32, __epi_m1);
            va = __builtin_epi_vload_2xf32(A[i], gvl);
            vb = __builtin_epi_vload_2xf32(B[i], gvl);
            vc = __builtin_epi_vfadd_2xf32(va, vb, gvl);
            __builtin_epi_vstore_2xf32(C[i], vc, gvl);
        }
    }

    return;
}

// Sum element-wise two matrices, now using vectorized instructions
void efficient_triad(float * A, float * B, float * C, int M, int N) {
    int i;
    int j;
    int gvl;
    __epi_2xf32 va;
    __epi_2xf32 vb;
    __epi_2xf32 vc;

    for (i = 0; i < M * N; i += gvl) {
        gvl = __builtin_epi_vsetvl(M * N - i, __epi_e32, __epi_m1);
        va = __builtin_epi_vload_2xf32( & A[i], gvl);
        vb = __builtin_epi_vload_2xf32( & B[i], gvl);
        vc = __builtin_epi_vfadd_2xf32(va, vb, gvl);
        __builtin_epi_vstore_2xf32( & C[i], vc, gvl);
    }

    return;
}

void scalar_relu(float * T, float * D, float alpha, int * shape, int N) {
    int i;
    int l = 1;
    for (i = 0; i < N; i++) l *= shape[i];
    for (i = 0; i < l; i++) D[i] = T[i] < 0 ? T[i] * alpha : T[i];

    return;
}

// Apply RELU function to all elements of a matrix
void relu(float * T, float * D, float alpha, int * shape, int N) {
    int i;
    int gvl;
    int l;
    __epi_2xf32 va;
    __epi_2xf32 v_zero;
    __epi_2xf32 v_alpha;
    __epi_2xf32 vc;
    __epi_2xi1 mask;

    // Multiply all dims to get tensor's flat shape
    l = 1;
    for (i = 0; i < N; i++) {
        l *= shape[i];
    }

    // Initialize auxiliar vectors
    v_zero = __builtin_epi_vbroadcast_2xf32(0, l);
    v_alpha = __builtin_epi_vbroadcast_2xf32(alpha, l);

    for (i = 0; i < l; i += gvl) {
        gvl = __builtin_epi_vsetvl(l - i, __epi_e32, __epi_m1);
        va = __builtin_epi_vload_2xf32( & T[i], gvl); // Load tensor
        mask = __builtin_epi_vmflt_2xf32(va, v_zero, gvl); // Create a 'less than' mask
        vc = __builtin_epi_vfmul_2xf32_mask(va, va, v_alpha, mask, gvl); // Multiply va by alpha only on elements specified by mask
        __builtin_epi_vstore_2xf32( & D[i], vc, gvl);
    }

    return;
}

void scalar_axpy(float * a, float * b, float * c, int N) {
    int i;
    * c = 0;
    for (i = 0; i < N; i++) {
        * c += a[i] * b[i];
    }

    return;
}

// Compute the inner product of two vectors
void axpy(float * a, float * b, float * c, int N) {
    int i;
    int gvl;
    __epi_2xf32 va;
    __epi_2xf32 vb;
    __epi_2xf32 vc;

    // Initialize first element of sum to zero
    __epi_2xf32 sum = __builtin_epi_vbroadcast_2xf32(0.0, 1);

    // Start inner products
    for (i = 0; i < N; i += gvl) {
        gvl = __builtin_epi_vsetvl(N - i, __epi_e32, __epi_m1);
        va = __builtin_epi_vload_2xf32( & a[i], gvl); // Load a
        vb = __builtin_epi_vload_2xf32( & b[i], gvl); // Load b
        vc = __builtin_epi_vfmul_2xf32(va, vb, gvl); // Multiply elementwise
        sum = __builtin_epi_vfredosum_2xf32(vc, sum, gvl); // Accumulate sum
    }

    // Store the first element of sum in c
    __builtin_epi_vstore_2xf32(c, sum, 1);

    return;
}

// Compute the EFFICIENT inner product of two vectors
void efficient_axpy(float * a, float * b, float * c, int N) {
    int i;
    int gvl;
    __epi_2xf32 va;
    __epi_2xf32 vb;

    // Initialize sum to zero and create zero_vec
    gvl = __builtin_epi_vsetvl(N, __epi_e32, __epi_m1);
    __epi_2xf32 sum = __builtin_epi_vbroadcast_2xf32(0.0, gvl);
    __epi_2xf32 zero_vec = __builtin_epi_vbroadcast_2xf32(0.0, gvl);

    // Start inner products
    // Note: we separate the last iteration, since tail agnostic functionality is not available
    for (i = 0; i < N / gvl; i++) {
        va = __builtin_epi_vload_2xf32( & a[i * gvl], gvl); // Load a
        vb = __builtin_epi_vload_2xf32( & b[i * gvl], gvl); // Load b
        sum = __builtin_epi_vfmacc_2xf32(sum, va, vb, gvl); // Accumulate products
    }

    // First reduction
    sum = __builtin_epi_vfredosum_2xf32(sum, zero_vec, gvl);

    if (N % gvl > 0) {
        va = __builtin_epi_vload_2xf32( & a[i * gvl], N % gvl); // Load a
        vb = __builtin_epi_vload_2xf32( & b[i * gvl], N % gvl); // Load b
        sum = __builtin_epi_vfmacc_2xf32(sum, va, vb, N % gvl); // Accumulate products

        // Second reduction
        sum = __builtin_epi_vfredosum_2xf32(sum, zero_vec, N % gvl);
    }

    __builtin_epi_vstore_2xf32(c, sum, 1);

    return;
}

// Sum element-wise a collection of tensors
void sum(float ** tensors, float * out, int * shape, int ndims, int ntensors) {
    int i;
    int j;
    int l;
    int gvl;
    __epi_2xf32 va;
    __epi_2xf32 sum;

    // Multiply all dims to get tensor's flat shape
    l = 1;
    for (i = 0; i < ndims; i++) {
        l *= shape[i];
    }

    // Iterate over the gvl-sized batches
    for (j = 0; j < l; j += gvl) {
        gvl = __builtin_epi_vsetvl(l - j, __epi_e32, __epi_m1);
        sum = __builtin_epi_vload_2xf32( & (tensors[0][j]), gvl); // Initialize to the first tensor

        // Iterate over the number of tensors
        for (i = 1; i < ntensors; i++) {
            va = __builtin_epi_vload_2xf32( & (tensors[i][j]), gvl);
            sum = __builtin_epi_vfadd_2xf32(va, sum, gvl);
        }

        // After the sum has been completed, store in out
        __builtin_epi_vstore_2xf32( & out[j], sum, gvl);
    }

    return;
}

// Unrolled sum element-wise a collection of tensors
void unrolled2_sum(float ** tensors, float * out, int * shape, int ndims, int ntensors) {
    int i;
    int j;
    int l;
    int gvl;
    __epi_2xf32 va1;
    __epi_2xf32 va2;
    __epi_2xf32 sum;

    // Multiply all dims to get tensor's flat shape
    l = 1;
    for (i = 0; i < ndims; i++) {
        l *= shape[i];
    }

    // Iterate over the gvl-sized batches
    for (j = 0; j < l; j += gvl) {
        gvl = __builtin_epi_vsetvl(l - j, __epi_e32, __epi_m1);
        sum = __builtin_epi_vbroadcast_2xf32(0.0, gvl); // Initialize to the first tensor

        // FACTOR 2
        for (i = 0; i < ntensors / 2; i += 2) {
            // Load tensors and accumulate (unrolled, factor 2)
            va1 = __builtin_epi_vload_2xf32( & (tensors[i][j]), gvl);
            va2 = __builtin_epi_vload_2xf32( & (tensors[i + 1][j]), gvl);

            // Accumulate using vectorized addition
            sum = __builtin_epi_vfadd_2xf32(va1, sum, gvl);
            sum = __builtin_epi_vfadd_2xf32(va2, sum, gvl);
        }

        // FACTOR 1
        for (; i < ntensors; i++) {
            va1 = __builtin_epi_vload_2xf32( & (tensors[i][j]), gvl);
            sum = __builtin_epi_vfadd_2xf32(va1, sum, gvl);
        }

        // After the sum has been completed, store in out
        __builtin_epi_vstore_2xf32( & out[j], sum, gvl);
    }

    return;
}