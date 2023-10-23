CC=clang
OPTS=-Ofast -march=native -flto 
CFLAGS=$(OPTS) -std=c11
LFLAGS=-flto 

CXX=clang++
CXXFLAGS=-I/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 $(OPTS) -std=c++17

matrix: matrix.o
matmult-eigen: matmult-eigen.o
clean:
	rm *.o
