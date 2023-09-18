#include <stdio.h>
#include <stdlib.h>

void vector_sum(float *, float *, float *, int);
void triad(float [][32], float [][32], float [][32], int, int);
void efficient_triad(float *, float *, float *, int, int);
void relu(float *, float *, float, int *, int);	
void axpy(float *, float *, float *, float, float, int);
void efficient_axpy(float *, float *, float *, int);
void sum(float **, float *, int *, int, int);
void unrolled2_sum(float **, float *, int *, int, int);
void unrolled4_sum(float **, float *, int *, int, int);

// Set the name of this function to main to run
int main_relu(void){
        int i;
        int j;
        int M=20;
        int N=20;
	int *shape = (int *)malloc(2 * sizeof(int));
        float alpha = 0.5;
	float *T = (float *)malloc(M * N * sizeof(float));
        float *D = (float *)malloc(M * N * sizeof(float));

	shape[0] = M; shape[1] = N;

        for(i=0; i<M*N; i++){
                T[i]=i-25;
        }

	printf("\n");
	printf("Initial tensor:\n");
        for(i=0; i<M; i++){
                for(j=0; j<N; j++){
                        printf("%.2f ", T[i*M+j]);
                }
                printf("\n");
        }
        printf("\n");

	relu(T, D, alpha, shape, 2);

	printf("\n");
	printf("Final tensor:\n");
        for(i=0; i<M; i++){
                for(j=0; j<N; j++){
                        printf("%.2f ", D[i*M+j]);
                }
                printf("\n");
        }
	printf("\n");

        return 0;
}

// Set the name of this function to main to run
int main_axpy(void){
	int i;
	int N = 2073;
	float *a = (float *)malloc(N * sizeof(float));
	float *b = (float *)malloc(N * sizeof(float));
	float c;
	float alpha = 1;
	float beta = 0;

	for(i=0; i<N; i++){
		a[i] = i;
		b[i] = 1;
	}

	int sum = 0;
	for(i=0; i<N; i++){
		sum += i;
		printf("%f %d\n", (float)i, sum);
	}
	printf("ATTENTION %d %d\n", sum, i);
	

	efficient_axpy(a, b, &c, N);

	printf("Result: %.2f\n", c);

	return 0;
}


// Set the name of this function to main to run
int main(void){
	int i;
	int j;
	int ndims = 1;
	int ntensors = 5;
	int *shape = (int *)malloc(ndims * sizeof(int));
	shape[0] = 129;

	float *out = (float *)malloc(shape[0] * sizeof(float));	// Shapes should be multiplied if more than one

	// Memory for tensors
	float **tensors = (float **)malloc(ntensors * sizeof(float *));
    	if (tensors == NULL) {
       		printf("Memory allocation failed\n");
       		return 1;
    	}
	for (int i = 0; i < ntensors; i++) {
		tensors[i] = (float *)malloc(shape[0] * sizeof(float));
		if (tensors[i] == NULL) {
			printf("Memory allocation failed\n");
       			return 1;
    		}
	}
	
	// Initialize tensors
	for(i=0; i<ntensors; i++){
		for(j=0; j<shape[0]; j++){
			tensors[i][j] = (i+1)*(j+1);
		}
	}
	
	// Sum of tensors
	sum(tensors, out, shape, ndims, ntensors);

	// Print result
	for(i=0; i<shape[0]; i++){
		printf("%.2f ", out[i]);
	}

	return 0;
}

// Sum element-wise two vectors using vectorized instructions
void vector_sum(float *a, float *b, float *c, int N){
        int i;
        int gvl;
        __epi_2xf32 va;
        __epi_2xf32 vb;
        __epi_2xf32 vc;

        for(i=0; i<N; i+=gvl){
                printf("i=%d\n", i);
		gvl = __builtin_epi_vsetvl(N-i, __epi_e32, __epi_m1);
                va = __builtin_epi_vload_2xf32(&a[i], gvl);
                vb = __builtin_epi_vload_2xf32(&b[i], gvl);
                vc = __builtin_epi_vfadd_2xf32(va, vb, gvl);
                __builtin_epi_vstore_2xf32(&c[i], vc, gvl);
        }
}

// Sum element-wise two matrices
void triad(float A[][32], float B[][32], float C[][32], int M, int N){
	int i;
	int j;
	int gvl;
	__epi_2xf32 va;
	__epi_2xf32 vb;
	__epi_2xf32 vc;

	for(i=0; i<M; i++){
		printf("i=%d\n", i);
		for(j=0; j<N; j+=gvl){
			printf("j=%d\n", j);
			gvl = __builtin_epi_vsetvl(N-j, __epi_e32, __epi_m1);
			va = __builtin_epi_vload_2xf32(A[i], gvl);
			vb = __builtin_epi_vload_2xf32(B[i], gvl);
			vc = __builtin_epi_vfadd_2xf32(va, vb, gvl);
			__builtin_epi_vstore_2xf32(C[i], vc, gvl);
		}
	}

	return;
}

// Sum element-wise two matrices, now using vectorized instructions
void efficient_triad(float *A, float *B, float *C, int M, int N){
        int i;
	int j;
        int gvl;
        __epi_2xf32 va;
        __epi_2xf32 vb;
        __epi_2xf32 vc;
	
        for(i=0; i<M*N; i+=gvl){
       		gvl = __builtin_epi_vsetvl(M*N-i, __epi_e32, __epi_m1);
                va = __builtin_epi_vload_2xf32(&A[i], gvl);
                vb = __builtin_epi_vload_2xf32(&B[i], gvl);
                vc = __builtin_epi_vfadd_2xf32(va, vb, gvl);
                __builtin_epi_vstore_2xf32(&C[i], vc, gvl);
        }

        return;
}

// Apply RELU function to all elements of a matrix
void relu(float *T, float *D, float alpha, int *shape, int N){
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
	for(i=0; i<N; i++){
		l*=shape[i];
	}

	// Initialize auxiliar vectors
	v_zero = __builtin_epi_vbroadcast_2xf32(0, l);
	v_alpha = __builtin_epi_vbroadcast_2xf32(alpha, l);

	for(i=0; i<l; i+=gvl){
		gvl = __builtin_epi_vsetvl(l-i, __epi_e32, __epi_m1);
		va = __builtin_epi_vload_2xf32(&T[i], gvl);				// Load tensor
		mask = __builtin_epi_vmflt_2xf32(va, v_zero, gvl);			// Create a 'less than' mask
		vc = __builtin_epi_vfmul_2xf32_mask(va, va, v_alpha, mask, gvl);	// Multiply va by alpha only on elements specified by mask
		__builtin_epi_vstore_2xf32(&D[i], vc, gvl);
	}

	return;
}

// Compute the inner product of two vectors
void axpy(float *a, float *b, float *c, float alpha, float beta, int N){
	int i;
	int gvl;
	float *sixte;
	__epi_2xf32 va;
        __epi_2xf32 vb;
        __epi_2xf32 vc;
	
	// Initialize first element of sum to zero
	__epi_2xf32 sum = __builtin_epi_vbroadcast_2xf32(0.0, 1);

	// Start inner products
	for(i=0; i<N; i+=gvl){
                gvl = __builtin_epi_vsetvl(N-i, __epi_e32, __epi_m1);
                va = __builtin_epi_vload_2xf32(&a[i], gvl);		// Load a
                vb = __builtin_epi_vload_2xf32(&b[i], gvl);		// Load b
                vc = __builtin_epi_vfmul_2xf32(va, vb, gvl);		// Multiply elementwise
		sum = __builtin_epi_vfredosum_2xf32(vc, sum, gvl);	// Accumulate sum
	}

	// Store the first element of sum in c
	__builtin_epi_vstore_2xf32(c, sum, 1);

	// Final scalar calculations
	*c = alpha * (*c) + beta;

	return;
}

// Compute the EFFICIENT inner product of two vectors (using FMA)
// This time we avoid the use of alpha and beta
void efficient_axpy(float *a, float *b, float *c, int N){
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
        for(i=0; i<N/gvl; i++){
		va = __builtin_epi_vload_2xf32(&a[i * gvl], gvl);       // Load a
                vb = __builtin_epi_vload_2xf32(&b[i * gvl], gvl);	// Load b
		sum = __builtin_epi_vfmacc_2xf32(sum, va, vb, gvl);	// Accumulate products
	}

	// First reduction
	sum = __builtin_epi_vfredosum_2xf32(sum, zero_vec, gvl);

	if(N%gvl > 0){
        	va = __builtin_epi_vload_2xf32(&a[i * gvl], N%gvl);       // Load a
        	vb = __builtin_epi_vload_2xf32(&b[i * gvl], N%gvl);       // Load b
        	sum = __builtin_epi_vfmacc_2xf32(sum, va, vb, N%gvl);     // Accumulate products

		// Second reduction
		sum = __builtin_epi_vfredosum_2xf32(sum, zero_vec, N%gvl);
	}

	__builtin_epi_vstore_2xf32(c, sum, 1);

        return;
}


// Sum element-wise a collection of tensors
void sum(float **tensors, float *out, int *shape, int ndims, int ntensors){
	int i;
	int j;
	int l;
	int gvl;
	__epi_2xf32 va;
	__epi_2xf32 sum;

	// Multiply all dims to get tensor's flat shape
	l = 1;
	for(i=0; i<ndims; i++){
		l*=shape[i];
       	}

	// Iterate over the gvl-sized batches
	for(j=0; j<l; j+=gvl){
		gvl = __builtin_epi_vsetvl(l-j, __epi_e32, __epi_m1);
		sum = __builtin_epi_vload_2xf32(&(tensors[0][j]), gvl);	// Initialize to the first tensor

		// Iterate over the number of tensors
		for(i=1; i<ntensors; i++){
                     	va = __builtin_epi_vload_2xf32(&(tensors[i][j]), gvl);
                     	sum = __builtin_epi_vfadd_2xf32(va, sum, gvl);
        	}

		// After the sum has been completed, store in out
		__builtin_epi_vstore_2xf32(&out[j], sum, gvl);
        }

        return;
}

// Unrolled sum element-wise a collection of tensors
void unrolled2_sum(float **tensors, float *out, int *shape, int ndims, int ntensors){
        int i;
        int j;
        int l;
        int gvl;
        __epi_2xf32 va1;
        __epi_2xf32 va2;
	__epi_2xf32 sum;

        // Multiply all dims to get tensor's flat shape
        l = 1;
        for(i=0; i<ndims; i++){
                l*=shape[i];
        }

        // Iterate over the gvl-sized batches
        for(j=0; j<l; j+=gvl){
                gvl = __builtin_epi_vsetvl(l-j, __epi_e32, __epi_m1);
                sum = __builtin_epi_vbroadcast_2xf32(0.0, gvl); // Initialize to the first tensor

		// FACTOR 2
		for (i = 0; i<ntensors/2; i += 2) {
    			// Load tensors and accumulate (unrolled, factor 2)
    			va1 = __builtin_epi_vload_2xf32(&(tensors[i][j]), gvl);
    			va2 = __builtin_epi_vload_2xf32(&(tensors[i + 1][j]), gvl);

    			// Accumulate using vectorized addition
 			sum = __builtin_epi_vfadd_2xf32(va1, sum, gvl);
    			sum = __builtin_epi_vfadd_2xf32(va2, sum, gvl);
		}

		// FACTOR 1
		for(; i<ntensors; i++){
                        va1 = __builtin_epi_vload_2xf32(&(tensors[i][j]), gvl);
                        sum = __builtin_epi_vfadd_2xf32(va1, sum, gvl);
                }

                // After the sum has been completed, store in out
                __builtin_epi_vstore_2xf32(&out[j], sum, gvl);
        }

        return;
} 

// Unrolled sum element-wise a collection of tensors
void unrolled4_sum(float **tensors, float *out, int *shape, int ndims, int ntensors){
        int i;
        int j;
        int l;
        int gvl;
        __epi_2xf32 va1;
        __epi_2xf32 va2;
	__epi_2xf32 va3;
        __epi_2xf32 va4;
        __epi_2xf32 sum;

        // Multiply all dims to get tensor's flat shape
        l = 1;
        for(i=0; i<ndims; i++){
                l*=shape[i];
        }

        // Iterate over the gvl-sized batches
        for(j=0; j<l; j+=gvl){
                gvl = __builtin_epi_vsetvl(l-j, __epi_e32, __epi_m1);
                sum = __builtin_epi_vbroadcast_2xf32(0.0, gvl);

		// FACTOR 4
                for (i = 0; i<ntensors/4; i += 4) {
                        va1 = __builtin_epi_vload_2xf32(&(tensors[i][j]), gvl);
                        va2 = __builtin_epi_vload_2xf32(&(tensors[i + 1][j]), gvl);
			va3 = __builtin_epi_vload_2xf32(&(tensors[i + 2][j]), gvl);
			va4 = __builtin_epi_vload_2xf32(&(tensors[i + 3][j]), gvl);

                        sum = __builtin_epi_vfadd_2xf32(va1, sum, gvl);
                        sum = __builtin_epi_vfadd_2xf32(va2, sum, gvl);
			sum = __builtin_epi_vfadd_2xf32(va3, sum, gvl);
                        sum = __builtin_epi_vfadd_2xf32(va4, sum, gvl);
                }

                // FACTOR 2
                for (; i<ntensors/2; i += 2) {
                        va1 = __builtin_epi_vload_2xf32(&(tensors[i][j]), gvl);
                        va2 = __builtin_epi_vload_2xf32(&(tensors[i + 1][j]), gvl);

                        sum = __builtin_epi_vfadd_2xf32(va1, sum, gvl);
                        sum = __builtin_epi_vfadd_2xf32(va2, sum, gvl);
                }

                // FACTOR 1
                for(; i<ntensors; i++){
                        va1 = __builtin_epi_vload_2xf32(&(tensors[i][j]), gvl);
                        sum = __builtin_epi_vfadd_2xf32(va1, sum, gvl);
                }

                // After the sum has been completed, store in out
                __builtin_epi_vstore_2xf32(&out[j], sum, gvl);
        }

        return;
}
