#include <iostream>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <random>

using namespace Eigen;

std::vector<long> gen_random_sample(long min, long max, long sample_size);
double get_random_double(double min, double max);
std::vector<double> get_vector_of_rn_doubles(int length, double min, double max);

int main(int argc, char *argv[])
{
  long N_COLS = 1024;
  long N_ROWS = 1024;
  long N_VALUES = 200000;

  if (argc == 2) 
    N_COLS = N_ROWS = strtol(argv[1], NULL, 0);
  else if (argc == 3) {
    N_ROWS = strtol(argv[1], NULL, 0); 
    N_COLS = strtol(argv[2], NULL, 0);
  } else if (argc == 4) {
    N_ROWS = strtol(argv[1], NULL, 0); 
    N_COLS = strtol(argv[2], NULL, 0);
    N_VALUES = strtol(argv[3], NULL, 0);
  }
  printf("r/c/v: %ld %ld %ld\n", N_ROWS, N_COLS, N_VALUES);

  SparseMatrix<double> big_A(N_ROWS, N_COLS);
  std::vector<long> cols_a = gen_random_sample(0, N_COLS, N_VALUES);
  std::vector<long> rows_a = gen_random_sample(0, N_ROWS, N_VALUES);
  std::vector<double> values_a = get_vector_of_rn_doubles(N_VALUES, 0, 1);

  for (int i = 0; i < N_VALUES; i++)
    big_A.coeffRef(rows_a[i], cols_a[i]) = values_a[i];
  // big_A.makeCompressed(); // slows things down

  SparseMatrix<double> big_B(N_ROWS, N_COLS);
  std::vector<long> cols_b = gen_random_sample(0, N_COLS, N_VALUES);
  std::vector<long> rows_b = gen_random_sample(0, N_ROWS, N_VALUES);
  std::vector<double> values_b = get_vector_of_rn_doubles(N_VALUES, 0, 1);

  for (int i = 0; i < N_VALUES; i++)
    big_B.coeffRef(rows_b[i], cols_b[i]) = values_b[i];
  // big_B.makeCompressed();

  SparseMatrix<double> big_AB(N_ROWS, N_COLS);

  clock_t begin = clock();

  big_AB = (big_A * big_B); //.pruned();

  clock_t end = clock();
  double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
  std::cout << "Time taken : " << elapsed_secs << std::endl;

}

int get_random_int(int min, int max)
{
  // std::uniform_real_distribution<int> dis(min, max);
  // std::default_random_engine re;
  // return dis(re);
  return (rand() % (max - min)) + min; 
}


std::vector<long> gen_random_sample(long min, long max, long sample_size)
{
  std::vector<long> my_vector(sample_size); // THE BUG, is right std::vector<long> my_vector

  for (long i = 0; i < sample_size; i++) {
    my_vector[i] = get_random_int((int)min, (int)max);
    // std::cout << my_vector[i] << std::endl;
  }
  return my_vector;
}

double get_random_double(double min, double max)
{
  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(min, max);
  double a_random_double = dis(gen);
  return a_random_double;
}

std::vector<double> get_vector_of_rn_doubles(int length, double min, double max)
{
  std::vector<double> my_vector(length);
  for (int i=0; i < length; i++) {
    my_vector[i] = get_random_double(min, max);
    // std::cout << my_vector[i] << std::endl;
  }
  return my_vector;
}