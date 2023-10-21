CXX=clang++
CC=clang
OPTS=-Ofast -march=native -flto 
CXXFLAGS=-I/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 $(OPTS) -std=c++17
CFLAGS=$(OPTS) -std=c11
LXXFLAGS=-flto 
LDFLAGS=-L/opt/homebrew/Cellar/eigen/3.4.0_1/include/eigen3 $(LXXFLAGS) -std=c++17

matrix: matrix.o
	$(CXX) $< -o $@
matmult-eigen: matmult-eigen.o
	$(CXX) $< -o $@
clean:
	rm *.o
