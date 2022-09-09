#include <Arduino.h>
#include <ArduinoEigen.h>

// .platformio/packages/framework-arduinoespressif32/tools/sdk/include/config/sdkconfig.h
// #define CONFIG_ARDUINO_LOOP_STACK_SIZE 16384

using namespace Eigen;


long N_COLS = 30;
long N_ROWS = 30;


// a naive matrix multiplication implementation. 
void matmult(double *A, double *B, double *C, int dimension)
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
void matmult_opt1(double *A, double *B, double *C, int dimension)
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
  long startTime, dur;
	double *A, *B, *C;

  long dimension = N_ROWS;
  long allocSize = dimension*dimension*sizeof(double);

#if 0
	A = (double*)heap_caps_malloc(allocSize, MALLOC_CAP_SPIRAM);
	B = (double*)heap_caps_malloc(allocSize, MALLOC_CAP_SPIRAM);
	C = (double*)heap_caps_malloc(allocSize, MALLOC_CAP_SPIRAM);
#else
	A = (double*)malloc(allocSize);
	B = (double*)malloc(allocSize);
	C = (double*)malloc(allocSize);
#endif
	// matrix initialization
	for(int i = 0; i < dimension; i++) {
		for(int j = 0; j < dimension; j++)
		{   
			A[dimension*i+j] = (rand()/(RAND_MAX + 1.0));
			B[dimension*i+j] = (rand()/(RAND_MAX + 1.0));
			C[dimension*i+j] = 0.0;
		}
	}  

  startTime = millis();
  matmult(A, B, C, dimension);
  dur = millis() - startTime;
  Serial.printf("%s (naive): Took %3ld ms. result=%.3f\r\n", __FUNCTION__, dur, C[0*dimension+0]);  

  memset(C, 0, allocSize);

  startTime = millis();
  matmult_opt1(A, B, C, dimension);
  dur = millis() - startTime;
  Serial.printf("%s (opt1): Took %3ld ms. result=%.3f\r\n", __FUNCTION__, dur, C[0*dimension+0]);  

	free(A);
	free(B);
	free(C);
}


void test_matmult_eigen()
{
  long startTime, dur;

  MatrixXd A = MatrixXd::Random(N_ROWS,N_COLS);
  MatrixXd B = MatrixXd::Random(N_ROWS,N_COLS);
  MatrixXd C = MatrixXd::Random(N_ROWS,N_COLS);

  SparseMatrix<double> big_A(N_ROWS, N_COLS);
  SparseMatrix<double> big_B(N_ROWS, N_COLS);
  SparseMatrix<double> big_C(N_ROWS, N_COLS);

  startTime = millis();
  C = A * B;
  dur = millis() - startTime;
  Serial.printf("%s (dense): Took %3ld ms. result=%.3f\r\n", __FUNCTION__, dur, C(0,0));  

  int n_entries = N_ROWS * N_COLS * 0.1; /* 10% */
  for (int i = 0; i < N_ROWS; i++) {
    for (int j = 0; j < N_COLS; j++) {
      big_A.coeffRef(i,j) = A(i,j);
      big_B.coeffRef(i,j) = B(i,j);
      if ((i * N_ROWS + j) % n_entries == 0) {
        startTime = millis();
        big_C = big_A * big_B;
        dur = millis() - startTime;
        Serial.printf("%s (sparse %ld): Took %3ld ms\r\n", 
          __FUNCTION__, (i * N_ROWS + j) / n_entries * 10, dur);  
      }
    }
  }
  startTime = millis();
  big_C = big_A * big_B;
  dur = millis() - startTime;
  Serial.printf("%s (sparse %d): Took %3ld ms. result=%.3f\r\n", __FUNCTION__, 100, dur, big_C.coeffRef(0,0));  
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);

  Serial.printf("Total heap: %d\n", ESP.getHeapSize());
  Serial.printf("Free heap: %d\n", ESP.getFreeHeap());
  Serial.printf("Total PSRAM: %d\n", ESP.getPsramSize());
  Serial.printf("Free PSRAM: %d\n", ESP.getFreePsram());

  test_matmult_mine();
  test_matmult_eigen();
}

void loop() {
  // put your main code here, to run repeatedly:
  
}