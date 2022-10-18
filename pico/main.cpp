#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#include <fcntl.h>
#include <sched.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/resource.h>

#include "pico/stdlib.h"

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>

using Eigen::MatrixXi;
using Eigen::SparseMatrix;


long N_COLS = 30;
long N_ROWS = 30;


// a naive matrix multiplication implementation. 
void matmult(int8_t *A, int8_t *B, int8_t *C, int dimension)
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
void matmult_opt1(int8_t *A, int8_t *B, int8_t *C, int dimension)
{
	for(int i = 0; i < dimension; i++) {
		for(int k = 0; k < dimension; k++) {
			for(int j = 0; j < dimension; j++) {
				C[dimension*i+j] += A[dimension*i+k] * B[dimension*k+j];
			}
		}
	}	
}

void test_matmult_mine()
{
  absolute_time_t start, dur;
	int8_t *A, *B, *C;

  long dimension = N_ROWS;
  long allocSize = dimension*dimension*sizeof(int8_t);

	A = (int8_t*)malloc(allocSize);
	B = (int8_t*)malloc(allocSize);
	C = (int8_t*)malloc(allocSize);

	// matrix initialization
	for(int i = 0; i < dimension; i++) {
		for(int j = 0; j < dimension; j++)
		{   
			A[dimension*i+j] = (rand()/(RAND_MAX + 1.0));
			B[dimension*i+j] = (rand()/(RAND_MAX + 1.0));
			C[dimension*i+j] = 0.0;
		}
	}  
  start = get_absolute_time();
  matmult(A, B, C, dimension);
  dur = get_absolute_time() - start;
  printf("%s (naive): Took %llu us. result=%i\r\n", __FUNCTION__, dur, C[0*dimension+0]);  

  memset(C, 0, allocSize);

  start = get_absolute_time();
  matmult_opt1(A, B, C, dimension);
  dur = get_absolute_time() - start;
  printf("%s (opt1): Took %llu us. result=%i\r\n", __FUNCTION__, dur, C[0*dimension+0]);  

	free(A);
	free(B);
	free(C);
}

void test_matmult_eigen()
{
  absolute_time_t start, dur;

  MatrixXi A = MatrixXi::Random(N_ROWS,N_COLS);
  MatrixXi B = MatrixXi::Random(N_ROWS,N_COLS);
  MatrixXi C = MatrixXi::Random(N_ROWS,N_COLS);

  SparseMatrix<int8_t> big_A(N_ROWS, N_COLS);
  SparseMatrix<int8_t> big_B(N_ROWS, N_COLS);
  SparseMatrix<int8_t> big_C(N_ROWS, N_COLS);

  start = get_absolute_time();
  C = A * B;
  dur = get_absolute_time() - start;
  printf("%s (dense): Took %llu us. result=%i\r\n", __FUNCTION__, dur, C(0,0));  

  for (int i = 0; i < N_ROWS; i++)
    for (int j = 0; j < N_COLS; j++) 
      big_A.coeffRef(i,j) = A(i,j);

  int n_entries = N_ROWS * N_COLS * 0.1; /* 10% */
  for (int i = 0; i < N_ROWS; i++) {
    for (int j = 0; j < N_COLS; j++) {
    retry:
      int i_idx = rand() % N_ROWS;
      int j_idx = rand() % N_COLS;

      if (B(i_idx, j_idx) == -99999)
        goto retry;
      big_B.coeffRef(i_idx,j_idx) = B(i_idx,j_idx);
      B(i_idx, j_idx) = -99999;
      
      if ((i * N_ROWS + j) % n_entries == 0) {
        start = get_absolute_time();
        big_C = big_A * big_B;
        dur = get_absolute_time() - start;
        printf("%s (sparse %ld): Took %llu us\r\n", 
          __FUNCTION__, (i * N_ROWS + j) / n_entries * 10, dur);  
      }
    }
  }
  start = get_absolute_time();
  big_C = big_A * big_B;
  dur = get_absolute_time() - start;
  printf("%s (sparse %d): Took %llu us. result=%i\r\n", __FUNCTION__, 100, dur, big_C.coeffRef(0,0));  
}

int main() {
  stdio_init_all();

  // printf("Total heap: %d\n", ESP.getHeapSize());
  // printf("Free heap: %d\n", ESP.getFreeHeap());
  // printf("Total PSRAM: %d\n", ESP.getPsramSize());
  // printf("Free PSRAM: %d\n", ESP.getFreePsram());
  sleep_ms(5000);
  test_matmult_mine();
  test_matmult_eigen();
}