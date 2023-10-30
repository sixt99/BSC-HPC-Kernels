
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

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
    for (i = 0; i < N; i++)
        l *= shape[i];

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
    for (i = 0; i < N; i++)
        * c += a[i] * b[i];

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

// Compute the OPTIMIZED inner product of two vectors
void optimized_axpy(float * a, float * b, float * c, int N) {
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

void scalar_sum(float ** tensors, float * out, int * shape, int ndims, int ntensors) {
    int i;
    int j;
    int l;
    float aux;

    // Multiply all dims to get tensor's flat shape
    l = 1;
    for (i = 0; i < ndims; i++) {
        l *= shape[i];
    }

    for (j = 0; j < l; j++){
        aux = 0;
        for (i = 0; i < ntensors; i++)
            aux += tensors[i][j];
        out[j] = aux;
    }

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

void scalar_dense(float * a, float * b, float * c, int * shape_a, int * shape_b){
    int i;
    int j;
    int k;
    int sum;
    int A = shape_a[0];
    int B = shape_a[1];
    int C = shape_b[1];

    for(i = 0; i<A; i++)
    for(j = 0; j<C; j++){
        sum = 0;
        for (k = 0; k < B; k++)
            sum += a[i * B + k] * b[k * C + j];
        c[i * C + j] = sum;
    }

    return;
}

void dense(float * a, float * b, float * c, int * shape_a, int * shape_b) {
    int i;
    int j;
    int k;
    int gvl;
    int A = shape_a[0];
    int B = shape_a[1];
    int C = shape_b[1];
    __epi_2xf32 va;
    __epi_2xf32 vb;

    // Transpose b
    float * b_transposed = (float * ) malloc(B*C * sizeof(float));
    for(i = 0; i < B; i++)
    for(j = 0; j < C; j++)
        b_transposed[j*B+i] = b[i*C+j];

    for(i = 0; i < A; i++){
        for(j = 0; j < C; j++){

            // Initialize sum to zero and create zero_vec
            gvl = __builtin_epi_vsetvl(B, __epi_e32, __epi_m1);
            __epi_2xf32 sum = __builtin_epi_vbroadcast_2xf32(0.0, gvl);
            __epi_2xf32 zero_vec = __builtin_epi_vbroadcast_2xf32(0.0, gvl);

            // Start inner products
            // Note: we separate the last iteration, since tail agnostic functionality is not available
            for (k = 0; k < B / gvl; k++) {
                va = __builtin_epi_vload_2xf32( a + i * B + k * gvl, gvl); // Load a
                vb = __builtin_epi_vload_2xf32( b_transposed + j * B + k * gvl, gvl); // Load b
                sum = __builtin_epi_vfmacc_2xf32(sum, va, vb, gvl); // Accumulate products
            }

            // First reduction
            sum = __builtin_epi_vfredosum_2xf32(sum, zero_vec, gvl);

            if (B % gvl > 0) {
                va = __builtin_epi_vload_2xf32( a + i * B + k * gvl, B % gvl); // Load a
                vb = __builtin_epi_vload_2xf32( b_transposed + j * B + k * gvl, B % gvl); // Load b
                sum = __builtin_epi_vfmacc_2xf32(sum, va, vb, B % gvl); // Accumulate products

                // Second reduction
                sum = __builtin_epi_vfredosum_2xf32(sum, zero_vec, B % gvl);
            }

            __builtin_epi_vstore_2xf32(c + A*i + j, sum, 1);

        }
    }

    return;
}

void dense2(float * a, float * b, float * c, int * shape_a, int * shape_b) {
    int i;
    int j;
    int k;
    int gvl;
    int A = shape_a[0];
    int B = shape_a[1];
    int C = shape_b[1];
    __epi_2xf32 va;
    __epi_2xf32 vb;

    for(i = 0; i < A; i++){
        for(j = 0; j < C; j++){

            // Initialize sum to zero and create zero_vec
            gvl = __builtin_epi_vsetvl(B, __epi_e32, __epi_m1);
            __epi_2xf32 sum = __builtin_epi_vbroadcast_2xf32(0.0, gvl);
            __epi_2xf32 zero_vec = __builtin_epi_vbroadcast_2xf32(0.0, gvl);

            // Start inner products
            // Note: we separate the last iteration, since tail agnostic functionality is not available
            for (k = 0; k < B / gvl; k++) {
                va = __builtin_epi_vload_2xf32( a + i * B + k * gvl, gvl); // Load a
                vb = __builtin_epi_vload_nt_strided_2xf32( b + C * k * gvl + j, C*4, gvl); // Load b
                sum = __builtin_epi_vfmacc_2xf32(sum, va, vb, gvl); // Accumulate products
            }

            // First reduction
            sum = __builtin_epi_vfredosum_2xf32(sum, zero_vec, gvl);

            if (B % gvl > 0) {
                va = __builtin_epi_vload_2xf32( a + i * B + k * gvl, B % gvl); // Load a
                vb = __builtin_epi_vload_nt_strided_2xf32( b + C * k * gvl + j, C*4, B % gvl); // Load b
                sum = __builtin_epi_vfmacc_2xf32(sum, va, vb, B % gvl); // Accumulate products

                // Second reduction
                sum = __builtin_epi_vfredosum_2xf32(sum, zero_vec, B % gvl);
            }

            __builtin_epi_vstore_2xf32(c + A*i + j, sum, 1);

        }
    }

    return;
}

/*
void scalar_softmax(float * T, float * D, int * shape, int ndims, int axis, int is_log){
    int i;
    int j;
    int stride;
    int sum;
    int l = 1;
    for (i = 0; i < ndims; i++)
        l *= shape[i];

    for (i = 0; i < l; i++)
        D[i] = exp(T[i]);

    // Compute the stride
    stride = 1;
    for (i = 0; i < axis; i++)
        stride *= shape[ndims-1-i];

    // Normalize according to the axis
    for (i = 0; i < l/shape[ndims-1-axis]; i++){
        sum = 0;
        for (j = 0; j < shape[ndims-1-axis]; j+=stride)
            sum += D[j];
        for (j = 0; j < shape[ndims-1-axis]; j+=stride)
            D[j]/=sum;
    }

    // Logsoftmax
    if(is_log)
        for (i=0; i<l; i++)
            D[i] = log(D[i]);

    return;
}
*/

void softmax(float * T, float * D, int * shape, int ndims, int axis, int is_log){
    
    #define vsetvl(avl)                 __builtin_epi_vsetvl(avl, __epi_e32, __epi_m1)
    #define vload(ptr)                  __builtin_epi_vload_2xf32(ptr, gvl)
    #define vstore(ptr, v)              __builtin_epi_vstore_2xf32(ptr, v, gvl)
    #define vfadd(a, b)                 __builtin_epi_vfadd_2xf32(a, b, gvl)
    #define vfmul(a, b)                 __builtin_epi_vfmul_2xf32(a, b, gvl)
    #define vmul_mask(a, b, c, mask)    __builtin_epi_vmul_2xi32_mask(a, b, c, mask, gvl)
    #define vfdiv(a, b)                 __builtin_epi_vfdiv_2xf32(a, b, gvl)
    #define vfdiv_mask(a, b, c, mask)   __builtin_epi_vfdiv_2xf32_mask(a, b, c, mask, gvl)
    #define vfbroadcast(a)              __builtin_epi_vbroadcast_2xf32(a, gvl)
    #define vbroadcast(a)               __builtin_epi_vbroadcast_2xi32(a, gvl)
    #define vf2i(a)                     __builtin_epi_vfcvt_x_f_2xi32_2xf32(a, gvl)
    #define vi2f(a)                     __builtin_epi_vfcvt_f_x_2xf32_2xi32(a, gvl)
    #define vfmacc(a, b, c)             __builtin_epi_vfmacc_2xf32(a, b, c, gvl)
    #define vmsgt(a, b)                 __builtin_epi_vmsgt_2xi32(a, b, gvl)
    #define vsll(a, b)                  __builtin_epi_vsll_2xi32(a, b, gvl)
    
    #define R_LN2f 1.442695040888963407359924681001892137426645954152985934135449406931f
    #define L2Uf 0.693145751953125f
    #define L2Lf 1.428606765330187045e-06f

    int i;
    int j;
    int stride;
    int gvl;
    int order;
    int l = 1;
    for (i = 0; i < ndims; i++)
        l *= shape[i];

    __epi_2xf32 d;
    __epi_2xf32 d_R_LN2f;
    __epi_2xf32 q;
    __epi_2xi32 integers;
    __epi_2xf32 s;
    __epi_2xf32 u;
    __epi_2xf32 tmp;
    __epi_2xf32 powers;
    __epi_2xi1 mask;

    // Apply the exponential function to every element
    for (i = 0; i < l; i += gvl) {
        gvl = vsetvl(l - i);
        d = vload(T + i);
        d_R_LN2f = vfmul(d, vfbroadcast(R_LN2f));   // Multiply d by 1/ln(2)
        integers = vf2i(d_R_LN2f);                  // Get the closest integer for each element in d_R_LN2f
        q = vi2f(integers);                         // Convert such integers to float again
        s = vfmacc(d, q, vfbroadcast(-L2Uf));
        s = vfmacc(s, q, vfbroadcast(-L2Lf));

        // Apply Taylor polynomial
        order = 5;
        u = vfbroadcast(1.0f);
        for(j = order; j>=1; --j){
            tmp = vfdiv(s, vfbroadcast(j));
            u = vfmacc(vfbroadcast(1.0f), tmp, u);
        }

        // Bit shift
        mask = vmsgt(vbroadcast(0), integers);                             // Create a 'smaller than zero' mask
        integers = vmul_mask(integers, integers, vbroadcast(-1), mask);    // Absolute value
        integers = vsll(vbroadcast(1), integers);                          // Powers of two        
        powers = vi2f(integers);                                           // To float
        powers = vfdiv_mask(powers, vfbroadcast(1.0f), powers, mask);      // Inverse of the negative values
        u = vfmul(u, powers);

        // Store
        vstore(D + i, u);
    }

    // Compute the stride
    stride = l;
    for (i = 0; i < axis+1; i++)
        stride /= shape[i];

    return;
}
