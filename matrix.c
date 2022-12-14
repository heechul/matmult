// https://vaibhaw-vipul.medium.com/matrix-multiplication-optimizing-the-code-from-6-hours-to-1-sec-70889d33dcfa

#define _GNU_SOURCE             /* See feature_test_macros(7) */
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

// a naive matrix multiplication implementation. 
void matmult(float *A, float *B, float *C, int dimension)
{
    for(int i = 0; i < dimension; i++) {
        for(int j = 0; j < dimension; j++) {
            for(int k = 0; k < dimension; k++) {
                C[dimension*i+j] += A[dimension*i+k] * B[dimension*k+j];
            }
        }
    }	
}

// a better cache optimized version: change the order to improve the cache hit rate
void matmult_opt1(float *A, float *B, float *C, int dimension)
{
    for(int i = 0; i < dimension; i++) {
        for(int k = 0; k < dimension; k++) {
            for(int j = 0; j < dimension; j++) {
                C[dimension*i+j] += A[dimension*i+k] * B[dimension*k+j];
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


void matmult_opt2(float *A, float *B, float *C, int dimension)
{
    int i,j,k;

    for(i = 0; i < dimension; i++) {
        for(j = 0; j < dimension; j++) {
            for(k = 0; k < dimension; k++) {                            
                C[dimension*i+j] += A[dimension*i+k] * B[dimension*j+k];
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
    int algo = 0;
    
    /*
     * get command line options 
     */
    while ((opt = getopt(argc, argv, "m:a:n:t:c:i:p:o:f:l:xh")) != -1) {
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
    Bt = (float*)malloc(alloc_size);
    C = (float*)malloc(alloc_size);
    memset(A, 0, alloc_size);
    memset(B, 0, alloc_size);
    memset(C, 0, alloc_size);
    memset(Bt, 0, alloc_size);
    
    srand(292);

    // init
    start = timestamp();
    for(i = 0; i < dimension; i++) {
        for(j = 0; j < dimension; j++) {
            A[dimension*i+j] = (rand()/(RAND_MAX + 1.0)) - 0.5;
            B[dimension*i+j] = (rand()/(RAND_MAX + 1.0)) - 0.5;
            C[dimension*i+j] = 0.0;
            // printf("%f %f\n", A[i*dimension+j], B[i*dimension+j]);
        }
    }
    end = timestamp();
    printf("init secs: %.6f\n", end-start);

    if (algo == 2) {
        // copy
        start = timestamp();    
        for(i = 0; i < dimension; i++) {
            for(j = 0; j < dimension; j++) {
                Bt[dimension*i+j] = B[dimension*i+j];
            }
        }
        end = timestamp();
        printf("Bt=B secs: %.6f\n", end-start);
    
        // do transpose
        start = timestamp();
        transpose_naive(B, Bt, dimension, dimension);
        end = timestamp();    
        printf("transpose secs: %.6f\n", end-start);
    }
    
    // do matrix multiplication
    start = timestamp();
    switch(algo) {
    case 0:
        matmult(A, B, C, dimension);
        break;
    case 1:
        matmult_opt1(A, B, C, dimension);
        break;
    case 2:
        matmult_opt2(A, Bt, C, dimension);
        break;
    }
    end = timestamp();
    printf("matmult secs: %.6f\n", end-start);
    
    // print sum
    float sum = 0; 
    for (i = 0; i < dimension; i++) {
        for (j = 0; j < dimension; j++) {
            sum += C[dimension * i + j];
        }
    }

    printf("C sums: %f\n", sum);
    free(A);
    free(B);
    free(C);
    
    return 0;
}
