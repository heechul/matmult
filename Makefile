CC=gcc
OPTS=-Ofast -march=native -flto
CFLAGS=$(OPTS) -std=c11 -g
LFLAGS=-flto 

CXX=g++
# Auto-detect Eigen path
EIGEN_PATH := $(shell if [ -d "/usr/include/eigen3" ]; then echo "/usr/include/eigen3"; elif [ -d "/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3" ]; then echo "/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3"; else echo "/usr/include/eigen3"; fi)
CXXFLAGS=-I$(EIGEN_PATH) $(OPTS) -std=c++17

all: matrix matmult-eigen-dense matmult-eigen-sparse

clean:
	rm -f matrix matmult-eigen-dense matmult-eigen-sparse
