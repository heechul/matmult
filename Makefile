CC=gcc
OPTS=-Ofast -march=native -flto 
CFLAGS=$(OPTS) -std=c11
LFLAGS=-flto 

CXX=g++
CXXFLAGS=-I/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 $(OPTS) -std=c++17

all: matrix matmult-eigen-dense matmult-eigen-sparse

clean:
	rm -f matrix matmult-eigen-dense matmult-eigen-sparse
