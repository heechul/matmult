// https://vaibhaw-vipul.medium.com/matrix-multiplication-optimizing-the-code-from-6-hours-to-1-sec-70889d33dcfa
// 
// g++ matrix.c -Ofast -march=native -flto


#ifndef _GNU_SOURCE
#  define _GNU_SOURCE             /* See feature_test_macros(7) */
#endif

#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>
#include <string.h>
// #include <omp.h>

/* change dimension size as needed */
struct timeval tv; 
int dimension = 1024;
double start, end; /* time */

double timestamp()
{
    double t;
    gettimeofday(&tv, NULL);
    t = tv.tv_sec + (tv.tv_usec/1000000.0);
    return t;
}

void init_data(float *A, float *B, float *C, int dimension)
{
    int i, j, k;
    srand(292);
    for(i = 0; i < dimension; i++) {
        for(j = 0; j < dimension; j++) {
            A[dimension*i+j] = (float)rand()/(float)(RAND_MAX) - 0.5;
            B[dimension*i+j] = (float)rand()/(float)(RAND_MAX) - 0.5;
            C[dimension*i+j] = 0.0;
        }
        // printf("%f %f\n", A[dimension*i+j], B[dimension*i+j]);
    }
}

double print_checksum(float *C, int dimention)
{
    double sum = 0.0;
    for(int i = 0; i < dimention; i++) {
        for(int j = 0; j < dimention; j++) {
            sum += C[i*dimention+j];
        }
    }
    return sum;
}

#define BENCH(func) \
    init_data(A, B, C, dimension); \
    start = timestamp(); \
    func; \
    end = timestamp(); \
    print_checksum(C, dimension); \
    printf(#func "  secs: %.6f  chsum: %.6f\n", end-start, print_checksum(C, dimension));


// a naive matrix multiplication implementation. 
void matmult_opt0_naive(float *A, float *B, float *C, int dimension)
{
    for(int i = 0; i < dimension; i++) {
        for(int j = 0; j < dimension; j++) {
            for(int k = 0; k < dimension; k++) {
                C[dimension*i+j] += (A[dimension*i+k] * B[dimension*k+j]);
            }
        }
    }	
}

// a better cache optimized version: change the order to improve the cache hit rate
void matmult_opt1_jk(float *A, float *B, float *C, int dimension)
{
    for(int i = 0; i < dimension; i++) {
        for(int k = 0; k < dimension; k++) {
            for(int j = 0; j < dimension; j++) {
                C[dimension*i+j] += (A[dimension*i+k] * B[dimension*k+j]);
            }
        }
    }	
}


// transposed
void transpose_naive(float *src, float *dst, int src_row, int src_col)
// src: m(src_row) x n(src_col)  -> dst: n x m
{
    for (int i = 0; i < src_col; i++) {
        for (int j = 0; j < src_row; j++) {
            dst[i*src_row+j] = src[j*src_col+i];
        }
    }
}

void matmult_opt2_transposed(float *A, float *B, float *C, int dimension)
{
    int i,j,k;
    int alloc_size = dimension*dimension*sizeof(float);
    float *Bt = (float*)malloc(alloc_size);
    transpose_naive(B, Bt, dimension, dimension);

    for(i = 0; i < dimension; i++) {
        for(j = 0; j < dimension; j++) {
            for(k = 0; k < dimension; k++) {                            
                C[dimension*i+j] += (A[dimension*i+k] * Bt[dimension*j+k]);
            }
        }
    }
    free(Bt);
}

#ifdef __SSE__
#include <emmintrin.h> // SSE2 Intrinsics
#include <smmintrin.h> // SSE4.2 Intrinsics

void matmult_opt3_transposed_simd(float* A, float* B, float* C, int dimension) {

    int alloc_size = dimension*dimension*sizeof(float);
    float *Bt = (float*)malloc(alloc_size);
    transpose_naive(B, Bt, dimension, dimension);

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            __m128 sum = _mm_setzero_ps(); // Initialize sum to zero

            for (int k = 0; k < dimension; k += 4) {
                // fprintf(stderr, "[%d,%d,%d]\n", i, j, k);
                __m128 a = _mm_load_ps(A + i * dimension + k); // Load 4 values from matrixA
                __m128 b = _mm_load_ps(Bt + j * dimension + k); // Load 4 values from matrixB
                __m128 mul = _mm_dp_ps(a, b, 0xF1); // Multiply and accumulate using dot product

                sum = _mm_add_ps(sum, mul);
                // Repeat the above steps for the remaining elements of the current row and column
            }

            // Store the result in the output matrix
            _mm_store_ss(C + i * dimension + j, sum);
            // fprintf(stderr, "[%d,%d]=%.2f\n", i, j, result[i*dimension+j]);
        }
    }
    free(Bt);
}
#elif __ARM_NEON
#include <arm_neon.h>
void matmult_opt3_transposed_simd(float* A, float* B, float* C, int dimension) {

    int alloc_size = dimension*dimension*sizeof(float);
    float *Bt = (float*)malloc(alloc_size);
    transpose_naive(B, Bt, dimension, dimension);

    // matrix multiplication of A and B into C
    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            float accumulators[4] = {0, 0, 0, 0};
            float32x4_t *acc = (float32x4_t *) accumulators;
            for (int k = 0; k < dimension; k += 4) {
                // fprintf(stderr, "[%d,%d,%d]\n", i, j, k);
                float32x4_t a = vld1q_f32(A + i * dimension + k); // Load 4 values from matrixA
                float32x4_t b = vld1q_f32(Bt + j * dimension + k); // Load 4 values from matrixB
                float32x4_t mul = vmulq_f32(a, b); // Multiply and accumulate using dot product
                *acc = vaddq_f32(*acc, mul);
                // Repeat the above steps for the remaining elements of the current row and column
            }
            // Store the result in the output matrix
            *(C + i * dimension + j) = accumulators[0] + accumulators[1] + accumulators[2] + accumulators[3];
            // fprintf(stderr, "[%d,%d]=%.2f\n", i, j, result[i*dimension+j]);
        }
    }
    free(Bt);
}
#endif // __SSE__ __ARM_NEON

// matrix multiplication with tiling    
void matmult_opt4_jk_tiling(float *A, float *B, float *C, int dimension)
{
    int i,j,k,ii,jj,kk;
    int bs = 32; // block size = 32*32*4 = 4KB

    for(i = 0; i < dimension; i+=bs) {
        for(k = 0; k < dimension; k+=bs) {
            for(j = 0; j < dimension; j+=bs) {
                for(ii = i; ii < i+bs; ii++) {
                    for(kk = k; kk < k+bs; kk++) {
                        for(jj = j; jj < j+bs; jj++) {
                            C[dimension*ii+jj] += (A[dimension*ii+kk] * B[dimension*kk+jj]);
                        }
                    }
                }
            }
        }
    }
}   

int main(int argc, char *argv[])
{
    float *A, *B, *Bt, *C;
    unsigned finish = 0;
    int i, j, k;
    
    int opt;
    int algo = 99;
    
    /*
     * get command line options 
     */
    while ((opt = getopt(argc, argv, "m:k:n:a:")) != -1) {
        switch (opt) {
        case 'n':
            dimension = strtol(optarg, NULL, 0);
            break;
        case 'a':
            algo = strtol(optarg, NULL, 0);
            break;
        }
    }
    
    printf("dimension: %d, algorithm: %d ws: %.1f\n", dimension, algo,
           (float)dimension*dimension*sizeof(float)*3/1024);

    int alloc_size = dimension*dimension*sizeof(float);
    A = (float*)malloc(alloc_size);
    B = (float*)malloc(alloc_size);
    C = (float*)malloc(alloc_size);
    memset(A, 0, alloc_size);
    memset(B, 0, alloc_size);
    memset(C, 0, alloc_size);
    
    // do matrix multiplication

    switch(algo) {
    case 0:
        BENCH(matmult_opt0_naive(A, B, C, dimension))
        break;
    case 1:
        BENCH(matmult_opt1_jk(A, B, C, dimension))
        break;
    case 2:
        BENCH(matmult_opt2_transposed(A, B, C, dimension))
        break;
    case 3:
        BENCH(matmult_opt3_transposed_simd(A, B, C, dimension))
        break;
    case 4:
        BENCH(matmult_opt4_jk_tiling(A, B, C, dimension))
        break;
    case 99:
        BENCH(matmult_opt0_naive(A, B, C, dimension))
        BENCH(matmult_opt1_jk(A, B, C, dimension))
        BENCH(matmult_opt2_transposed(A, B, C, dimension))
        BENCH(matmult_opt3_transposed_simd(A, B, C, dimension))
        BENCH(matmult_opt4_jk_tiling(A, B, C, dimension))
        break;
    default:
        printf("invalid algorithm\n");
        break;
    }
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}
