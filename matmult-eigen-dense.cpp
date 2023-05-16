#include <iostream>
#include <Eigen/Dense>
#include <time.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

using namespace std;

uint64_t get_elapsed(struct timespec *start, struct timespec *end)
{
	uint64_t dur;

	dur = ((uint64_t)end->tv_sec * 1000000000 + end->tv_nsec) - 
		((uint64_t)start->tv_sec * 1000000000 + start->tv_nsec);
	return dur;
}

int main(int argc, char *argv[])
{
  long N_COLS = 1024;
  long N_ROWS = 1024;

  struct timespec start, end;

  if (argc == 2) 
    N_COLS = N_ROWS = strtol(argv[1], NULL, 0);
  else if (argc == 3) {
    N_ROWS = strtol(argv[1], NULL, 0); 
    N_COLS = strtol(argv[2], NULL, 0);
  }
  printf("r/c: %ld %ld\n", N_ROWS, N_COLS);
  
  MatrixXd A = MatrixXd::Random(N_ROWS,N_COLS);
  MatrixXd B = MatrixXd::Random(N_ROWS,N_COLS);
  MatrixXd C = MatrixXd::Random(N_ROWS,N_COLS);

  clock_gettime(CLOCK_REALTIME, &start);
  C = A * B;
  clock_gettime(CLOCK_REALTIME, &end);

//   std::cout << "A =" << std::endl << A << std::endl;
//   std::cout << "B =" << std::endl << B << std::endl;
//   std::cout << "C =" << std::endl << C << std::endl;

  printf("Init took %.6f s\n", (double) get_elapsed(&start, &end)/1000000000);

}