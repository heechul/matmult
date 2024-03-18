#include <Arduino.h>

// .platformio/packages/framework-arduinoespressif32/tools/sdk/include/config/sdkconfig.h
// #define CONFIG_ARDUINO_LOOP_STACK_SIZE 16384

struct timeval tv; 
double start, end;
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
    printf("%.12s  secs: %.6f  chsum: %.6f\n", #func, end-start, print_checksum(C, dimension));


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

// matrix multiplication with jk order switch
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

// matrix multiplication with jk order switch and tiling    
void matmult_opt2_jk_tiling(float *A, float *B, float *C, int dimension)
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


// transpose matrix
void transpose_naive(float *src, float *dst, int src_row, int src_col)
// src: m(src_row) x n(src_col)  -> dst: n x m
{
    for (int i = 0; i < src_col; i++) {
        for (int j = 0; j < src_row; j++) {
            dst[i*src_row+j] = src[j*src_col+i];
        }
    }
}

// matrix multiplicaiton after transposed
void matmult_opt3_transposed(float *A, float *B, float *C, int dimension)
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

// matrix multiplicaiton transposed with SIMD
void matmult_opt4_transposed_simd(float* A, float* B, float* C, int dimension) {

    int alloc_size = dimension*dimension*sizeof(float);
    float *Bt = (float*)malloc(alloc_size);
    transpose_naive(B, Bt, dimension, dimension);

    for (int i = 0; i < dimension; i++) {
        for (int j = 0; j < dimension; j++) {
            float accumulators[4] = {0, 0, 0, 0};
            __m128 *acc = (__m128 *) accumulators;
            for (int k = 0; k < dimension; k += 4) {
                // fprintf(stderr, "[%d,%d,%d]\n", i, j, k);
                __m128 a = _mm_load_ps(A + i * dimension + k); // Load 4 values from matrixA
                __m128 b = _mm_load_ps(Bt + j * dimension + k); // Load 4 values from matrixB
                __m128 mul = _mm_mul_ps(a, b); // Multiply and accumulate using dot product
                *acc = _mm_add_ps(*acc, mul);
                // Repeat the above steps for the remaining elements of the current row and column
            }
            // Store the result in the output matrix
            *(C + i * dimension + j) = accumulators[0] + accumulators[1] + accumulators[2] + accumulators[3];
            // fprintf(stderr, "[%d,%d]=%.2f\n", i, j, result[i*dimension+j]);
        }
    }
    free(Bt);
}
#elif __ARM_NEON
#include <arm_neon.h>
// matrix multiplicaiton transposed with SIMD
void matmult_opt4_transposed_simd(float* A, float* B, float* C, int dimension) {

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
#else
void matmult_opt4_transposed_simd(float* A, float* B, float* C, int dimension) {
    fprintf(stderr, "SIMD is not supported\n");
}
#endif // __SSE__ __ARM_NEON


int bench_matmult(int dimension, int algo)
{
    float *A, *B, *Bt, *C;
    unsigned finish = 0;
    int i, j, k;

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
        BENCH(matmult_opt2_jk_tiling(A, B, C, dimension))
        break;
    case 3:
        BENCH(matmult_opt3_transposed(A, B, C, dimension))
        break;
    case 4:
        BENCH(matmult_opt4_transposed_simd(A, B, C, dimension))
        break;
    case 99:
        BENCH(matmult_opt0_naive(A, B, C, dimension))
        BENCH(matmult_opt1_jk(A, B, C, dimension))
        BENCH(matmult_opt2_jk_tiling(A, B, C, dimension))
        BENCH(matmult_opt3_transposed(A, B, C, dimension))
        BENCH(matmult_opt4_transposed_simd(A, B, C, dimension))
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

void setup() {
    int algo = 99;
    int dimension = 64;

    // put your setup code here, to run once:
    Serial.begin(115200);

    while(!Serial) {
        static int retries = 0;
        delay(1000); // Wait for serial monitor to open
        if (retries++ > 5) {
        break;
        }
    } // When the serial monitor is turned on, the program starts to execute

    Serial.printf("Total heap: %d\n", ESP.getHeapSize());
    Serial.printf("Free heap: %d\n", ESP.getFreeHeap());
    Serial.printf("Total PSRAM: %d\n", ESP.getPsramSize());
    Serial.printf("Free PSRAM: %d\n", ESP.getFreePsram());

    bench_matmult(dimension, algo);
}

void loop() {
  // put your main code here, to run repeatedly: 
}